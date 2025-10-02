import collections
import os
import json
import hashlib
import typing as t

import openai as oai

import config.configuration as cfg
import oai_batch

class MergeSimilarProductsWorkItem(oai_batch.WorkItem):
    def __init__(self, unified_product_name: str, products_to_merge: t.List[str], product_info: t.Dict[str, t.Dict[str, str]], model_name: str, work_dir: str):
        super().__init__()
        self._unified_product_name = unified_product_name
        self._products_to_merge = products_to_merge
        self._product_info = product_info
        self._model_name = model_name
        self._work_dir = work_dir
        self._response_data: t.Optional[t.Dict[str, str]] = None
        os.makedirs(work_dir, exist_ok=True)

        prods = "||".join(products_to_merge)

        self._item_id = hashlib.md5(";;;".join([prods, model_name]).encode()).hexdigest()

    def get_id(self) -> str:
        return self._item_id

    def _get_request_body(self, request_id: str, prompt: str, input: str, model_name: str) -> str:
        body = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": input}
                ]
            }
        }
        return json.dumps(body, ensure_ascii=False)

    def _get_merge_prompt_instructions(self) -> str:
        instructions = """Your task is to summarise the description of following products. They are closely related 
and differs mostly in colours and dimensions. In descriptions find common properties and ignore the differences in 
colour and dimensions. Keep only essential product characteristics. If products are too diverse, output empty response.

Products are characterised by their PRODUCT_LABEL, SHORT_DESCRIPTION, LONG_DESCRIPTION, SUMMARY and PRODUCT_TAGS.
PRODUCT_LABEL is name of the product
SHORT_DESCRIPTION and LONG_DESCRIPTION are marketing description taken from the manufacturer's website
SUMMARY is generated description from all available product information
PRODUCT_TAGS is a JSON-like structure capturing design, area of use, load capacity as the system infer them.
Individual products consists of all four information and are separated by hashes ########################

{PRODUCT_LABELS}

########################
END OF PRODUCT LIST

Output only the summary description if the products are consistent enough or empty string if they are too diverse.
Write all texts in Czech.
"""
        return instructions

    def _get_product_descriptions(self, products_to_merge: t.List[str], product_info: t.Dict[str, t.Dict[str, str]]) -> str:
        prod_desc_list = []
        for prod_merge in products_to_merge:
            prod_nfo = product_info[prod_merge]
            prod_str = f"""
########################
PRODUCT_LABEL: '{prod_nfo["product_title"]}' 
SHORT_DESCRIPTION: '{prod_nfo["product_short_description"]}'
LONG_DESCRIPTION: '{prod_nfo["product_details_description"]}'
SUMMARY: '{prod_nfo["product_summary"]}'
PRODUCT_TAGS: '{prod_nfo["product_classifications"]}'
"""
            prod_desc_list.append(prod_str)
        return "\n".join(prod_desc_list)

    def get_jsonl_list(self) -> t.List[str]:
        instruction_str = self._get_merge_prompt_instructions()
        prod_desc = self._get_product_descriptions(products_to_merge=self._products_to_merge, product_info=self._product_info)
        instruction_str = instruction_str.format(PRODUCT_LABELS=prod_desc)

        batch_input = [
            self._get_request_body(request_id="product_summary", model_name=self._model_name, prompt=instruction_str, input="The summary is:"),
        ]

        return batch_input

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"merged_products-{self.get_id()}.json")

    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

    def save_resposes(self, response_jsons: t.List[str]) -> None:
        all_colours = set()
        for prod in self._products_to_merge:
            col = self._product_info[prod]["product_param_colour"]
            bar = self._product_info[prod]["product_parameters"]["BARVA"] if "BARVA" in self._product_info[prod]["product_parameters"] else "NA"
            if col != bar:
                print("XXXX")
            all_colours.add(col)

        summary_text = response_jsons[0]["response"]["body"]["choices"][0]["message"]["content"]
        store_object = {
            "base_product_name": self._unified_product_name,
            "products_to_merge": self._products_to_merge,
            "grouped_summary": summary_text,
            "all_colours": list(all_colours),
        }

        self._response_data = store_object

        with open(self._get_output_filename(), "wt", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )

    def get_responses(self) -> t.Dict[str, str]:
        if not self.is_new():
            with open(self._get_output_filename(), "rt", encoding="utf-8") as f:
                return json.load(fp=f)
        else:
            raise Exception("Not yet evaluated")

    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        pass

class MakeProductGroupsWorkItem(oai_batch.WorkItem):
    def __init__(self, product_descriptions: t.List[t.Dict[str, str]],  model_name: str, work_dir: str):
        super().__init__()
        self._product_descriptions = product_descriptions
        self._model_name = model_name
        self._work_dir = work_dir
        self._response_data: t.Optional[t.Dict[str, str]] = None
        os.makedirs(work_dir, exist_ok=True)

        prods = "||".join([v["AH PRODUCT GROUP"] for v in product_descriptions])

        self._item_id = hashlib.md5(";;;".join([prods, model_name]).encode()).hexdigest()

    def get_id(self) -> str:
        return self._item_id

    def _get_request_body(self, request_id: str, prompt: str, input: str, model_name: str) -> str:
        body = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": input}
                ]
            }
        }
        return json.dumps(body, ensure_ascii=False)

    def _get_merge_group_instructions(self) -> str:
        instructions = """
Based on the description of the product groups below, create a product group of products with similar: characteristics, puropose, 
intended use, load bearing capability (pedestrians only, car, trucks) - ie. garden tiles, paving, construction, kerb elements, etc.
 
For each product there is PRODUCT NAME and DESCRIPTION available. Products are separated by hashes ########################.

The list of products follows:

{PRODUCTS}

################################################
### END OF PRODUCTS 
 
For each product group, oupput a name capturing the core characteristics, one-paragraph description capturing 
characteristics of products within the group and the list of products assigned into that product group. Make sure, 
all products are present in the output and all products are assigned to exactly one product group. All output
must be in Czech. The output has to be a JSON with following structure:
{{
    "PRODUCT GROUPS": [
        {{
            "PRODUCT GROUP NAME": "<name>",
            "PRODUCT GROUP DESCRIPTION": "<description>",
            "PRODUCT LIST": ["<product 1>", "<product 2>", ...]
        }}
    ]
}}"""
        return instructions

    def _get_product_descriptions(self, product_descriptions: t.List[t.Dict[str, str]]) -> str:
        prod_desc_list = []
        for prod in product_descriptions:
            prod_str = f"""
########################
PRODUCT NAME: '{prod["unified_product"]}' 
DESCRIPTION: '{prod["unified_summary"]}'
"""
            prod_desc_list.append(prod_str)
        return "\n".join(prod_desc_list)

    def get_jsonl_list(self) -> t.List[str]:
        instruction_str = self._get_merge_group_instructions()
        prod_desc = self._get_product_descriptions(product_descriptions=self._product_descriptions)
        instruction_str = instruction_str.format(PRODUCTS=prod_desc)
        prompt_str = "The product groups are:"

        batch_input = [
            self._get_request_body(request_id="alt_prod_hierarchy", model_name=self._model_name, prompt=instruction_str, input=prompt_str),
        ]

        return batch_input

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"alternative_hierarchy-{self.get_id()}.json")

    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

    def save_resposes(self, response_jsons: t.List[str]) -> None:
        summary_text = response_jsons[0]["response"]["body"]["choices"][0]["message"]["content"]
        store_object = {
            "product_groups": summary_text
        }

        self._response_data = store_object

        with open(self._get_output_filename(), "wt", encoding="utf-8") as f:
            json.dump(
                obj=store_object,
                fp=f,
                indent=4,
                ensure_ascii=False
            )

    def get_responses(self) -> t.Dict[str, str]:
        if not self.is_new():
            with open(self._get_output_filename(), "rt", encoding="utf-8") as f:
                return json.load(fp=f)
        else:
            raise Exception("Not yet evaluated")

    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        pass

if __name__ == '__main__':
    with open(os.path.join(cfg.get_settings().main_results_path_dir, "all_products.json"), "r", encoding="utf-8") as f:
        product_info = json.load(fp=f)
    with open(os.path.join(cfg.get_settings().main_results_path_dir, "base_product_titles.json"), "r", encoding="utf-8") as f:
        product_groups_mapping = json.load(fp=f)

    product_db = {}
    for pi in product_info:
        product_db[pi["product_title"]] = pi

    oai_batch = oai_batch.OAI_Batch(
        client=oai.Client(api_key=cfg.get_settings().open_ai_api_key),
        number_active_batches=200,
        max_work_items_to_add=100,
        return_work_items=False,
        working_dir=cfg.get_settings().product_grouping_request_cache_dir
    )

    work_items = []
    for product_group_name, product_names in product_groups_mapping.items():
        wi = MergeSimilarProductsWorkItem(
            products_to_merge=product_names,
            unified_product_name=product_group_name,
            product_info=product_db,
            model_name=cfg.get_settings().product_summary_extraction_model_name,
            work_dir=cfg.get_settings().product_grouping_request_cache_dir
        )
        work_items.append(wi)

    oai_batch.add_work_items(work_items)
    oai_batch.run_loop()

    product_group_2_product_mapping: t.List[t.Dict[str, str]] = []
    for wi in work_items:
        resp = wi.get_responses()
        product_group_2_product_mapping.append({
            "AH PRODUCT GROUP": resp["base_product_name"],
            "AH PRODUCT": resp["products_to_merge"],
            "AH PRODUCT GROUP DESCRIPTION": resp["grouped_summary"],
            "AH PRODUCT ALL COLOURS": resp["all_colours"]
        })

    product_group_db = {}
    for up in product_group_2_product_mapping:
        product_group_db[up["AH PRODUCT GROUP"]] = up

    wi = MakeProductGroupsWorkItem(product_descriptions=product_group_2_product_mapping, model_name=cfg.get_settings().hierarchy_inference_model_name, work_dir=cfg.get_settings().product_grouping_request_cache_dir)
    oai_batch.add_work_items([wi])
    oai_batch.run_loop()

    product_types_json = wi.get_responses()
    product_types_mapping = json.loads(product_types_json["product_groups"])["PRODUCT GROUPS"]
    product_types_db = {}

    missed_product_groups = set()
    halucinated_product_groups = set()
    product_groups_mentioned = collections.Counter()

    for prod_grp_nm in list(product_group_db.keys()):
        missed_product_groups.add(prod_grp_nm)

    for pt in product_types_mapping:
        all_colours = set()
        for prod_name in pt["PRODUCT LIST"]:
            all_colours.update(product_group_db[prod_name]["AH PRODUCT ALL COLOURS"])

        product_types_db[pt["PRODUCT GROUP NAME"]] = {
            "AH PRODUCT TYPE NAME": pt["PRODUCT GROUP NAME"],
            "AH PRODUCT TYPE DESCRIPTION": pt["PRODUCT GROUP DESCRIPTION"],
            "AH PRODUCT GROUPS": pt["PRODUCT LIST"],
            "AH PRODUCT TYPE COLOURS": list(all_colours)
        }
        for prod in pt["PRODUCT LIST"]:
            product_groups_mentioned[prod] += 1
            if prod in product_group_db:
                missed_product_groups.remove(prod)
            else:
                halucinated_product_groups.add(prod)

    print(missed_product_groups)
    print(halucinated_product_groups)
    print(product_groups_mentioned)

    alternative_hierarchy_db = {
        "product_db": product_db,
        "product_groups_mapping": product_groups_mapping,
        "product_group_db": product_group_db,
        "product_types_db": product_types_db
    }

    with open(os.path.join(cfg.get_settings().main_results_path_dir, "alternative_hierarchy_db.json"), "wt", encoding="utf-8") as f:
        json.dump(obj=alternative_hierarchy_db, fp=f, indent=4, ensure_ascii=False)


