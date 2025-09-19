import json
import typing as t
import tqdm
import os
import hashlib
import openai as oai

import oai_batch

class SumWorkItem(oai_batch.WorkItem):
    def __init__(self, group_name: str, prompt: str, items: t.List[str], model_name: str, work_dir: str):
        super().__init__()
        self._group_name = group_name
        os.makedirs(work_dir, exist_ok=True)
        content = [prompt]
        content.extend(items)
        self._prompt = prompt
        self._items = items
        self._item_id = hashlib.md5(";;;".join(content).encode()).hexdigest()
        self._work_dir = work_dir
        self._model_name = model_name

    def get_id(self):
        return self._item_id

    def _get_request_body(self, request_id: str, conversation: t.List, model_name: str) -> str:
        body = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": conversation
            }
        }
        return json.dumps(body, ensure_ascii=False)

    def get_jsonl_list(self) -> t.List[str]:
        message = []
        message.append({"role": "developer", "content": self._prompt})
        for i in self._items:
            message.append({"role": "user", "content": i})
        request = self._get_request_body(request_id=self.get_id(), conversation=message, model_name=self._model_name)
        return [request]

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"hierarchy-summary-{self.get_id()}.json")

    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

    def save_resposes(self, response_jsons: t.List[str]) -> None:
        summary_text = response_jsons[0]["response"]["body"]["choices"][0]["message"]["content"]
        gn = self._group_name
        store_object = {
            "group_name": gn,
            "summary_text": summary_text
        }
        with open(self._get_output_filename(), "w", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )

    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        pass

    def get_response(self) -> str:
        with open(self._get_output_filename(), "rt", encoding="utf-8") as f:
            return f.read()



def do_product_group_summary(prod_group_2_product: t.Dict[str, t.Union[t.List[str], t.Set[str]]], prod_group_desc: t.Dict[str, str], description_per_product: t.Dict[str, t.Dict[str, str]], workdir: str, model_name: str) -> t.Dict[str, str]:
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    batcher = oai_batch.OAI_Batch(client=client, working_dir=workdir, number_active_batches=200, max_work_items_to_add=110, return_work_items=False)

    pg_summary: t.Dict[str, str] = {}
    prompt = """In the following messages, there are product descriptions of individual products in product group '{pg_name}'. Create a short text summarising 
the products (and their properties) in the product group. Focus on what the products have in common - ie. function, intended use, shapes, etc. The text will be 
later used to match products to items enquired by a potential customer. Customer enquires are fuzzy and can be misaligned. To help you, the first message 
contains the marketing description of the product group."""

    for pg, prods in prod_group_2_product.items():
        pg_desc = prod_group_desc[pg]
        all_product_summaries = [
            f"Product group '{pg}' marketing description: {pg_desc if pg_desc != '' else 'No description provided'}",
        ]
        prods = prods.copy()
        prods.sort()
        for prod in prods:
            all_product_summaries.append(f"{prod}: {description_per_product[prod]['product_summary']}")

        swi = SumWorkItem(group_name=pg, prompt=prompt, items=all_product_summaries, work_dir=workdir, model_name=model_name)
        batcher.add_work_items(work_items=[swi])

    batcher.run_loop()

    prod_group_summary: t.Dict[str, str] = {}
    for filename in list(filter(lambda fn: fn.endswith("json") and fn.startswith("hierarchy-summary"), os.listdir(workdir))):
        full_fn: str = os.path.join(workdir, filename)
        with open(full_fn, "r", encoding="utf-8") as f:
            sum_obj = json.load(fp=f)
            prod_group_summary[sum_obj["group_name"]] = sum_obj["summary_text"]

    return prod_group_summary


def do_product_family_summary(prod_family_2_prod_group: t.Dict[str, t.List[str]], prod_family_description: t.Dict[str, str], description_per_product_group: t.Dict[str, str], summary_per_product_group: t.Dict[str, str], workdir: str, model_name: str) -> t.Dict[str, str]:
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    batcher = oai_batch.OAI_Batch(client=client, working_dir=workdir, number_active_batches=200, max_work_items_to_add=110, return_work_items=False)

    prompt = """Your task is to come up with a summary text of one level in product hierarchy, specifically you have to come up 
with summary of specific product family. Product family consists of several product groups, each in turn consists of several products.
The aim is to get summary capturing general product characteristics within the product family. In the summary focus on type of products
their characteristics, intended use, function, shapes, etc. The text will be later used to match products to items 
enquired by a potential customer. Customer enquires are fuzzy and can be misaligned. To help you, the first message contains the marketing
description of the product family.
 
The input consists of: first, the name and then marketing description of the product family itself; second, the name the marketing
description of individual product groups; and thirdly summary description generated from individual products."""

    for pf, prodgrps in prod_family_2_prod_group.items():
        pf_desc = prod_family_description[pf]
        lines = [
            f"Product family '{pf}' marketing description: {pf_desc if pf_desc != '' else 'No description provided'}",
        ]
        prodgrps = prodgrps.copy()
        prodgrps.sort()
        for pgrp in prodgrps:
            lines.append(f"Marketing description of product group '{pgrp}': '{description_per_product_group[pgrp]}'")
        for pgrp in prodgrps:
            lines.append(f"Summary obtained from products in the product group '{pgrp}': '{summary_per_product_group[pgrp]}'")

        swi = SumWorkItem(group_name=pf, prompt=prompt, items=lines, work_dir=workdir, model_name=model_name)
        batcher.add_work_items(work_items=[swi])

    batcher.run_loop()

    prod_family_summary: t.Dict[str, str] = {}
    for filename in list(filter(lambda fn: fn.endswith("json") and fn.startswith("hierarchy-summary"), os.listdir(workdir))):
        full_fn: str = os.path.join(workdir, filename)
        with open(full_fn, "r", encoding="utf-8") as f:
            sum_obj = json.load(fp=f)
            prod_family_summary[sum_obj["group_name"]] = sum_obj["summary_text"]

    return prod_family_summary

def do_category_summary(category_2_prod_family: t.Dict[str, t.List[str]], category_desc: t.Dict[str, str], description_per_product_family: t.Dict[str, str], summary_per_product_family: t.Dict[str, str], workdir: str, model_name: str):
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    batcher = oai_batch.OAI_Batch(client=client, working_dir=workdir, number_active_batches=200, max_work_items_to_add=110, return_work_items=False)

    prompt = """Your task is to come up with a summary text of one level in product hierarchy, specifically you have to come up 
with summary of specific category. Category consists of several product families, each in turn consists of several product groups.
The aim is to get summary capturing general product characteristics within the category. In the summary focus on type of products
their characteristics, intended use, function, shapes, etc. The text will be later used to match products to items 
enquired by a potential customer. Customer enquires are fuzzy and can be misaligned. To help you, the first message contains the marketing
description of the category.
 
The input consists
 of: first, the name and then marketing description of the category itself; second, the name and the marketing
description of individual product families; and thirdly summary description generated from individual product groups within
the scpeficif product family."""

    for cat, prod_fams in category_2_prod_family.items():
        cat_desc = category_desc[cat]
        lines = [
            f"Category '{cat}' marketing description: {cat_desc if cat_desc != '' else 'No description provided'}",
        ]
        prod_fams = prod_fams.copy()
        prod_fams.sort()
        for pgrp in prod_fams:
            lines.append(f"Marketing description of product family '{pgrp}': '{description_per_product_family[pgrp]}'")
        for pgrp in prod_fams:
            lines.append(f"Summary obtained from products in the product family '{pgrp}': '{summary_per_product_family[pgrp]}'")

        swi = SumWorkItem(group_name=cat, prompt=prompt, items=lines, work_dir=workdir, model_name=model_name)
        batcher.add_work_items(work_items=[swi])

    batcher.run_loop()

    prod_family_summary: t.Dict[str, str] = {}
    for filename in list(filter(lambda fn: fn.endswith("json") and fn.startswith("hierarchy-summary"), os.listdir(workdir))):
        full_fn: str = os.path.join(workdir, filename)
        with open(full_fn, "r", encoding="utf-8") as f:
            sum_obj = json.load(fp=f)
            prod_family_summary[sum_obj["group_name"]] = sum_obj["summary_text"]

    return prod_family_summary

def main():
    with open('all_products.json') as f:
        product_data: t.List[t.Dict[str, t.Any]] = json.load(f)

    description_per_category: t.Dict[str, str] = {}
    description_per_product_family: t.Dict[str, str] = {}
    description_per_product_group: t.Dict[str, str] = {}
    description_per_product: t.Dict[str, t.Dict[str, str]] = {}

    category_2_prod_family: t.Dict[str, t.Union[t.List[str], t.Set[str]]] = {}
    prod_family_2_category: t.Dict[str, str] = {}

    prod_family_2_prod_group: t.Dict[str, t.Union[t.List[str], t.Set[str]]] = {}
    prod_group_2_prod_family: t.Dict[str, str] = {}

    prod_group_2_product: t.Dict[str, t.Union[t.List[str], t.Set[str]]] = {}
    product_2_prod_group: t.Dict[str, str] = {}

    for pd in tqdm.tqdm(product_data):
        if pd['category'] not in description_per_category:
            description_per_category[pd['category']] = pd['category_description']
        if pd['prod_family'] not in description_per_product_family:
            description_per_product_family[pd['prod_family']] = pd['prod_family_description']
        if (pd['product_group'] != "") and (pd['product_group'] not in description_per_product_group):
            description_per_product_group[pd['product_group']] = pd['product_group_description']

        if pd['category'] not in category_2_prod_family:
            category_2_prod_family[pd['category']] = set()
        category_2_prod_family[pd['category']].add(pd['prod_family'])
        prod_family_2_category[pd['prod_family']] = pd['category']

        if pd['prod_family'] not in prod_family_2_prod_group:
            prod_family_2_prod_group[pd['prod_family']] = set()
        prod_family_2_prod_group[pd['prod_family']].add(pd['product_group'])
        prod_group_2_prod_family[pd['product_group']] = pd['prod_family']

        if pd['product_group'] not in prod_group_2_product:
            prod_group_2_product[pd['product_group']] = set()
        prod_group_2_product[pd['product_group']].add(pd['product_title'])
        product_2_prod_group[pd['product_title']] = pd['product_group']

        print(pd['product_title'])
        if pd['product_title'] in ["BEST - BASE DRENO ANTRACITOVÁ", "BEST - BEATON PŮLKA PŘÍRODNÍ", "BEST - URIKO I PŘÍRODNÍ", "BEST - KROSO PŘÍRODNÍ"]:
            print(pd['product_classifications'])
            pd['product_classifications'] = pd['product_classifications'].replace("```", "").replace("json", "")

        dims = json.loads(pd["product_dimensions"])
        prod_classifications = json.loads(pd["product_classifications"])

        description_per_product[pd["product_title"]] = {
            "title": pd['product_title'],

            "product_classifications": prod_classifications,
            "product_details_description": pd['product_details_description'],
            "product_dimensions": dims,
            "product_parameters": pd["product_parameters"],
            "product_short_description": pd['product_short_description'],
            "product_summary": pd["product_summary"],
            "product_summary_hierarchy": pd["product_summary_hierarchy"],
            "product_table_details": pd["product_table_details"],
            "product_url": pd['product_url'],
        }

    index = {}
    for k, v in category_2_prod_family.items():
        index[k] = list(v)
    category_2_prod_family = index

    index = {}
    for k, v in prod_family_2_prod_group.items():
        index[k] = list(v)
    prod_family_2_prod_group = index

    index = {}
    for k, v in prod_group_2_product.items():
        index[k] = list(v)
    prod_group_2_product = index

    # os.makedirs("/home/cepekmir/Sources/wdf_best_catalog/tmp/batches", exist_ok=True)
    product_group_summary = {}
    product_family_summary = {}
    category_summary = {}

    product_group_summary: t.Dict[str, str] = do_product_group_summary(
        prod_group_2_product=prod_group_2_product,
        prod_group_desc=description_per_product_group,
        description_per_product=description_per_product,
        workdir="/home/cepekmir/Sources/wdf_best_catalog/tmp/prod_group_batches",
        model_name="gpt-5-mini"
    )

    product_family_summary: t.Dict[str, str] = do_product_family_summary(
        prod_family_2_prod_group=prod_family_2_prod_group,
        prod_family_description=description_per_product_family,
        description_per_product_group=description_per_product_group,
        summary_per_product_group=product_group_summary,
        workdir="/home/cepekmir/Sources/wdf_best_catalog/tmp/prod_family_batches",
        model_name="gpt-5-mini"
    )

    category_summary: t.Dict[str, str] = do_category_summary(
        category_2_prod_family=category_2_prod_family,
        category_desc=description_per_category,
        description_per_product_family=description_per_product_family,
        summary_per_product_family=product_family_summary,
        workdir="/home/cepekmir/Sources/wdf_best_catalog/tmp/category_batches",
        model_name="gpt-5-mini"
    )


    big_index_thingy = {
        "description_per_category": description_per_category,
        "description_per_product_family": description_per_product_family,
        "description_per_product_group": description_per_product_group,
        "description_per_product": description_per_product,
        "category_2_prod_family": category_2_prod_family,
        "prod_family_2_category": prod_family_2_category,
        "prod_family_2_prod_group": prod_family_2_prod_group,
        "prod_group_2_prod_family": prod_group_2_prod_family,
        "prod_group_2_product": prod_group_2_product,
        "product_2_prod_group": product_2_prod_group,
        "prod_group_summary": product_group_summary,
        "prod_family_summary": product_family_summary,
        "category_summary": category_summary,
    }

    with open('all_product_indices.json', 'w', encoding="utf-8") as f:
        json.dump(big_index_thingy, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()