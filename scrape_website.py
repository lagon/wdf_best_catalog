import concurrent.futures as futures
import hashlib
import io
import itertools
import json
import os
import time
import traceback
import typing as t
import random

import pydantic as pyd

import bs4
from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.parse as urlparse
import tqdm
import openai as oai
from resources.template_codes import TEMPLATE_CODES


class ProductDim(pyd.BaseModel):
    height: int
    length: int
    thickness: int

class ProductDimensions(pyd.BaseModel):
    dimensions: t.List[ProductDim]


def make_batch(client: oai.OpenAI, batch_input: t.List[t.Dict]) -> str:
    content_lines = []
    for bi in batch_input:
        content_lines.append(json.dumps(bi))
    jsonl_content = "\n".join(content_lines)

    oai_file = client.files.create(file=io.BytesIO(jsonl_content.encode("utf-8")), purpose="batch") #, expires_after={"anchor": "created_at", "seconds": 90000})

    batch = client.batches.create(
        input_file_id=oai_file.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    return batch.id

def product_web_extraction_prompts(model_name: str, product_detail_desc: str) -> t.List:
    """
    Create a batch job with two requests (description extraction + table extraction).
    Returns the batch id.
    """
    batch_input = [
        {
            "custom_id": "description",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {"role": "developer", "content": """Remove html tags from following part of the web page. Keep basic formatting, like paragraphs or lists and bullet points, replace those with stars. Also remove all the table data from the page. Make sure all outputs are in Czech."""},
                    {"role": "user", "content": product_detail_desc}
                ],
            },
        },
        {
            "custom_id": "table",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {"role": "developer", "content": """The following piece of the HTML contains one or more HMTL tables containing product name, it's dimensions and packaging. Extract the content of the table(s) in row-order. Precede each cell in the table with the column name and each row in the table on the single line. Extract only name of the product name and it's dimensions (skladebné rozměry). Omit duplicate lines.  Make sure all outputs are in Czech."""},
                    {"role": "user", "content": product_detail_desc}
                ],
            },
        },
        # {
        #     "custom_id": "table",
        #     "method": "POST",
        #     "url": "/v1/responses",
        #     "body": {
        #         "model": model_name,
        #         "input": [
        #             {"role": "developer", "content": """The following piece of the HTML contains various parameters product parameters - typically colour (barva), finish (povrch) and height (výška) of the product, but possibly others as well. Part the    Make sure all outputs are in Czech."""},
        #             {"role": "user", "content": product_parameters}
        #         ],
        #     }
        # }
    ]
    return batch_input

def wait_for_batch(client: oai.OpenAI, batch_id: str, batch_poll_interval_secs: int) -> oai.types.Batch:
    """
    Poll until the batch job completes or fails. Returns the batch object.
    """
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"TICK-TOCK {batch.id} status {batch.status}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(batch_poll_interval_secs)


def parse_batch_results(client: oai.OpenAI, batch: oai.types.Batch) -> t.Dict[str, str]:
    """
    Download and parse the batch output. Returns description_text, table_text.
    """
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} did not produce an output file.")

    file_response = client.files.content(batch.output_file_id)
    file_content = file_response.read().decode("utf-8")
    results = [json.loads(line) for line in file_content.strip().splitlines()]

    out_data: t.Dict[str, str] = {}
    for res in results:
        custom_id = res.get("custom_id")
        output_text = res["response"]["body"]["output"][0]["content"][0]["text"]
        out_data[custom_id] = output_text

    return out_data


def run_batch_with_retry(client: oai.OpenAI, batch_data: t.List, max_batch_retries: int, batch_poll_interval_secs: int) -> t.Optional[t.Dict[str, str]]:
    """
    Run a batch with retry logic for failed/expired/cancelled batches.
    """
    for attempt in range(1, max_batch_retries + 1):
        print("Sending out batches")
        batch_id = make_batch(client=client, batch_input=batch_data)
        print(f"Waiting for batch {batch_id} to complete")
        batch = wait_for_batch(client, batch_id, batch_poll_interval_secs)

        print(f"Batch {batch_id} is finished with status: {batch.status}")
        if batch.status == "completed":
            return parse_batch_results(client, batch)

        print(f"⚠️ Batch {batch_id} failed with status={batch.status}, attempt {attempt}/{max_batch_retries}")

        if attempt == max_batch_retries:
            raise RuntimeError(f"Batch failed after {max_batch_retries} attempts (last status={batch.status}).")

        time.sleep(2 * attempt)  # backoff before retry
    return None


def extract_text_description_and_table(product_title: str, product_detail_desc: str) -> t.Optional[t.Tuple[str, str]]:
    """
    Main function: cache results locally, otherwise run batch job (with retry) and return results.
    """
    # Cache filename
    fname = os.path.join(
        "/home/cepekmir/Sources/wdf_best_catalog/tmp",
        f"content-{hashlib.md5(";;;".join([product_title, product_detail_desc]).encode()).hexdigest()}.json"
    )

    model_name = "gpt-4.1-mini"
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))

    # Return from cache if available
    if os.path.isfile(fname):
        with open(fname, "rt", encoding="utf-8") as f:
            cont = json.load(f)
            return cont["description"], cont["table"]

    # Run batch with retry
    batch_data = product_web_extraction_prompts(model_name=model_name, product_detail_desc=product_detail_desc)
    batch_res = run_batch_with_retry(client=client, batch_data=batch_data, max_batch_retries=3, batch_poll_interval_secs=10)
    if batch_res is None:
        return None

    # Save to cache
    with open(fname, "wt", encoding="utf-8") as f:
        json.dump(batch_res, f, ensure_ascii=False, indent=4)

    return batch_res["description"], batch_res["table"]

def summary_batch_requests(model_name: str,
                           product_name: str,
                           short_desc: str,
                           long_desc: str,
                           table: str,
                           prod_group_desc: str,
                           prod_family_desc: str,
                           category_desc: str) -> t.List[t.Dict]:
    """
    Create a batch job with two requests (description extraction + table extraction).
    Returns the batch id.
    """
    batch_input = [
        {
            "custom_id": "product_summary",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {"role": "developer", "content": """Your task is summarise a description of a building product made of pressed concrete. 
The summary must capture the intended use (garden, home, public roads) and purpose, essential characteristics like  dimensions and colour. 
If the description captures different flavours of the product, mention them all in the summary. The text will be later used to match 
products to items enquired by a potential customer. Customer enquires are fuzzy and can be misaligned. So keep all relevant details 
in the description. The product description is in czech and consists of four elements: product name, short description (marketing paragraph), longer
description (more technical description) and table with dimensions and details in row order. Make sure, you mention the product name in the summary.
Keep all replies in Czech language."""},
                    {"role": "user", "content": f"Product name: {product_name}"},
                    {"role": "user", "content": f"Short description marketing headline description: {short_desc}"},
                    {"role": "user", "content": f"Longer, more technical description: {long_desc}"},
                    {"role": "user", "content": f"Table with dimensions and details in row order, with column names preceding the values {table}."},
                    {"role": "user", "content": f"The summary:"}
                ],
            },
        },
        {
            "custom_id": "hierarchy_summary",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {"role": "developer", "content": """Your task is summarise a description of a building product made of pressed concrete.
The summary must capture the intended use (garden, home, public roads) and purpose, essential characteristics like  dimensions and colour. 
If the description captures different flavours of the product, mention them all in the summary. The text will be later used to match 
products to items enquired by a potential customer. Customer enquires are fuzzy and can be misaligned. So keep all relevant details 
in the description. The product description is in czech and consists of four elements: product name, short description (marketing paragraph), longer
description (more technical description) and table with dimensions and details in row order. In addition you get also all the descriptions
from product hierarchy - from the top, category (most general set of products), product family and product group (lowest level).  Keep all replies in Czech language."""},
                    {"role": "user", "content": f"Product name: {product_name}"},
                    {"role": "user", "content": f"Short description marketing headline description: {short_desc}"},
                    {"role": "user", "content": f"Longer, more technical description: {long_desc}"},
                    {"role": "user", "content": f"Table with dimensions and details in row order, with column names preceding the values {table}."},
                    {"role": "user", "content": f"Product group hierarchy: {prod_group_desc}."},
                    {"role": "user", "content": f"Product family hierarchy: {prod_family_desc}."},
                    {"role": "user", "content": f"Category hierarchy: {category_desc}."},
                    {"role": "user", "content": f"The summary:"}
                ],
            },
        },
        {
            "custom_id": "classifications",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {"role": "developer", "content": """Your task is classify the product by it's design (for example: zámková dlažba,
dlažba pro slepce, potrubí, obrubník, etc)., area of use (for example: zahrada, chodníky, kanalizace), load capacity (for example: není zmíněna, 
pochozí, pojezd osobních automobilů, pojezd nákladních automobilů), finish (for example: mrazuvzdorná, není mrazuvzdorná, 
odolná proti rozmrazovacím prostředkům) and technical norms mentioned (like: 'ČSN EN 206-1', 'XF4').
If some of the information is not mentioned or not obvoious, produce empty list.
Output the is JSON with following structure:
{
    'design': ['zámková dlažba'],
    'area of use': ['zahrada'],
    'load capacity': ['pochozí'],
    'finish': ['mrazuvzdorná'],
    'technical norms': ['XF4']
}
Keep all replies in Czech language."""},
                    {"role": "user", "content": f"Product name: {product_name}"},
                    {"role": "user", "content": f"Short description marketing headline description: {short_desc}"},
                    {"role": "user", "content": f"Product group hierarchy: {prod_group_desc}."},
                    {"role": "user", "content": f"The summary:"}
                ],
            },
        },
        {
            "custom_id": "dimensions",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {"role": "developer", "content": """Your task identify dimensions of a product. The product's dimensions
are shown in the row-wise pseudo-tabular form with column names just before the values. Give just a list of product dimensions
in JSON format - the example follows: [{'label': 'BEST - AKVAGRAS', 'width': 10, 'length': 50, 'height': 20}]. Only return valid JSON, no extra text and 
keep only the fields in the example above, do not use any other. Do not prefix the output with json. The column mapping from the HTML table to the fields above typically is:
label = název; width = šířka, tloušťka or D; length = délka or L and height = výška or t."""},
                    {"role": "user", "content": f"Table with dimensions and details in row order, with column names preceding the values: {table}."},
                    {"role": "user", "content": f"Dimensions:"}
                ],
            },
        }
    ]
    return batch_input

def get_product_summaries_and_tags(prod_name: str, short_desc: str, long_desc: str, table: str, prod_group_desc: str, prod_family_desc: str, category_desc: str) -> t.Optional[t.Dict[str, str]]:
    model_name = "gpt-4.1-mini"
    prod_desc_digest=hashlib.md5(";;;".join([model_name, prod_name, short_desc, long_desc, table, prod_group_desc, prod_family_desc, category_desc]).encode()).hexdigest()
    fname=os.path.join("/home/cepekmir/Sources/wdf_best_catalog/tmp", f"summary-{prod_desc_digest}.json")
    if os.path.isfile(fname):
        with open(fname, mode="rt", encoding="utf-8") as f:
            return json.load(f)

    client: oai.OpenAI = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    batch_data = summary_batch_requests(model_name=model_name,
                                        product_name=prod_name,
                                        short_desc=short_desc,
                                        long_desc=long_desc,
                                        table=table,
                                        prod_group_desc=prod_group_desc,
                                        prod_family_desc=prod_family_desc,
                                        category_desc=category_desc)

    batch_res = run_batch_with_retry(client=client, batch_data=batch_data, max_batch_retries=3, batch_poll_interval_secs=10)
    if batch_res is None:
        return None

    # Save to cache
    with open(fname, "wt", encoding="utf-8") as f:
        json.dump(batch_res, f, ensure_ascii=False, indent=4)

    return batch_res

def download_website(url: str) -> t.Optional[str]:
    cache_path = os.path.join("/home/cepekmir/Sources/wdf_best_catalog/tmp/web/", hashlib.md5(url.encode()).hexdigest())
    if os.path.exists(cache_path):
        with open(cache_path, "rt") as f:
            return f.read()
    else:
        response = requests.get(url)
        if response.status_code == 200:
            with open(cache_path, "wt") as f:
                f.write(response.text)
            return response.text
        else:
            return None

def extract_links_from_category_page(webpage: str, web_url: str, base_url: str) -> t.Dict[str, t.Union[t.List[str], str]]:
    # Parse with BeautifulSoup
    soup = BeautifulSoup(webpage, "html.parser")

    # Extract all links from them
    links = []
    category_title = soup.find(name="div", class_="supercategory__left").find(name="h1").text
    category_description = soup.find(name="div", class_="supercategory__left--description").text.strip()

    # Find all divs with class 'card-product'
    card_divs = soup.find_all("div", class_="supercategory__whitebox--banners d-flex")
    for div in card_divs:
        for a in div.find_all("a", href=True):
            href = a["href"]
            href = urlparse.urljoin(base_url, href)
            links.append(href)

    # Remove duplicates if needed
    links = list(set(links))

    return {
        "category": category_title,
        "category_description": category_description,
        "category_url": web_url,
        "links": links
    }

def extract_links_from_prod_family_page(webpage: str, web_url: str, base_url: str) -> t.Dict[str, t.Union[t.List[str], str]]:
    soup = BeautifulSoup(webpage, "html.parser")
    # Extract all links from them
    links = set()

    prod_fam_div = soup.find(name="div", class_="category__left")
    if prod_fam_div is None:
        return {}

    prod_family_title = prod_fam_div.find(name="h1").text
    prod_family_description = soup.find(name="div", class_="category__left--description").text.strip()

    prod_family_list = soup.find(name="div", id="category-list__content")
    for link in prod_family_list.find_all("a", href=True):
        links.add(link["href"])

    return {
        "prod_family": prod_family_title,
        "prod_family_description": prod_family_description,
        "prod_family_url": web_url,
        "prod_family_links": list(links)
    }

def is_product_page(webpage: str) -> bool:
    soup = BeautifulSoup(webpage, "html.parser")
    category_list_grid = soup.find(name="div", id="category-list__content")
    return category_list_grid is None

def extract_products_from_product_groups(webpage: str, web_url, base_url: str) -> t.List[t.Dict[str, t.Any]]:
    soup = BeautifulSoup(webpage, "html.parser")
    category_list_content_list = soup.find(name="div", id="category-list__content", class_="product-grid")
    product_cards = category_list_content_list.find_all(name="div", class_="card-product")
    prod_subgroup_title = soup.find(name="div", class_="category__left").find(name="h1").text
    prod_subgroup_description = soup.find(name="div", class_="category__left--description").text.strip()

    products = []
    for card in product_cards:
        product_link = card.find(name="a", href=True)["href"]
        if product_link == web_url:
            print("Self link!")
        else:
            products.append({
                "product_group": prod_subgroup_title,
                "product_group_description": prod_subgroup_description,
                "product_group_url": web_url,
                "product_url": product_link,
                "is_product": False
            })
    return products

def extract_details_from_prod_page(webpage: str, web_url: str) -> t.Dict[str, t.Union[t.List[str], str]]:
    soup = BeautifulSoup(webpage, "html.parser")
    # Extract all links from them
    # print(web_url)

    if web_url.split("/")[-1] in TEMPLATE_CODES:
        template_codes = [web_url.split("/")[-1]]
    else:
        template_codes = ["NA"] #Chce to vic prace

    prod_detail_div = soup.find(name="div", class_="product-detail__header")

    product_title = prod_detail_div.find(name="h1").text
    product_short_description = soup.find(name="div", class_="product-detail-cart__text").text.strip()
    product_long_description = soup.find(name="div", class_="product-detail-description__content").text.strip()
    product_parameters_element = soup.find(name="div", class_="product-detail-cart__form product-detail-cart__form--params")

    prod_param_elems_flag = [type(s) is bs4.Tag for s in list(product_parameters_element)]
    product_parameters_element = list(itertools.compress(list(product_parameters_element), prod_param_elems_flag))

    product_paramters: t.Dict[str, t.Union[str, t.List[str]]] = {}
    i = 0
    while i < len(product_parameters_element):
        label: str = product_parameters_element[i].text.strip()
        label = label.replace(":", "")
        value_tag: bs4.Tag = product_parameters_element[i + 1]
        if value_tag.name == "select":
            value = [txt.text for txt in list(value_tag.find_all(name="option"))]
        else:
            value = product_parameters_element[i + 1].text.strip()
        product_paramters[label] = value
        i = i + 2

    product_details_text, product_details_table = extract_text_description_and_table(product_title=product_title, product_detail_desc=product_long_description)

    return {
        "product_title": product_title,
        "product_short_description": product_short_description,
        "product_details_description": product_details_text,
        "product_table_details": product_details_table,
        "product_parameters": product_paramters,
        "product_templates": template_codes
    }

def _row_to_formated_cells(row, table_tag):
    headers = []
    for hdr in row.find_all(name=table_tag):
        colspan = 1
        rowspan = 1
        if "colspan" in hdr.attrs:
            colspan = int(hdr.attrs["colspan"])
        if "rowspan" in hdr.attrs:
            rowspan = int(hdr.attrs["rowspan"])

        for _ in range(colspan):
            h_info = {"name": hdr.text.strip(), "rowspan": rowspan}
            headers.append(h_info)
    return headers

def _regularise_table(content_rows):
    if len(content_rows) == 0:
        return  []
    for col_id in range(len(content_rows[0])):
        for row_id in range(len(content_rows)):
            if row_id+1 >= len(content_rows):
                continue
            if content_rows[row_id][col_id]["rowspan"] > 1:
                content_rows[row_id + 1].insert(col_id, {
                    "name": content_rows[row_id][col_id]["name"],
                    "rowspan": content_rows[row_id][col_id]["rowspan"] - 1,
                })
                content_rows[row_id][col_id]["rowspan"] = 1
    return content_rows

def _join_labels(l1: str, l2: str) -> str:
    if l1 == l2:
        return l1
    if l1 == "":
        return l2
    if l2 == "":
        return l1
    return f"{l1} & {l2}"

def process_one_product(prod: t.Dict[str, t.Any]) -> t.Tuple[t.List, t.List]:
    time.sleep(random.uniform(0,2))
    link = prod["product_url"]
    # print(link)
    webtxt = download_website(link)
    if webtxt is None:
        print(f"   !!!! Failed to download {link} in product group {prod['product_group_url']}.  Skipping.")
        return [], []

    if not prod["is_product"]:
        rex_prods = []
        if is_product_page(webpage=webtxt):
            product_dict = {
                "product_url": link,
                "product_group": prod["product_group"] if "product_group" in prod else "NA",
                "product_group_description": prod["product_group_description"] if "product_group_description" in prod else "",
                "product_group_url": link,
                "is_product": True
            }
            rex_prods.append(product_dict)
        else:
            rex_prods.extend(extract_products_from_product_groups(webpage=webtxt, web_url=link, base_url="https://www.best.cz/"))
        for rp in rex_prods:
            rp["category"] = prod["category"]
            rp["category_description"] = prod["category_description"]
            rp["category_url"] = prod["category_url"]
            rp["prod_family"] = prod["prod_family"]
            rp["prod_family_description"] = prod["prod_family_description"]
            rp["prod_family_url"] = prod["prod_family_url"]
            # rp["product_url"] = link
            rp["product_group"] = _join_labels(prod["product_group"], rp["product_group"])
            rp["product_group_description"] = _join_labels(prod["product_group_description"], rp["product_group_description"])
            rp["product_group_url"] = link
        return rex_prods, []
    else:
        try:
            prod_details = extract_details_from_prod_page(webpage=webtxt, web_url=link)
            if len(prod) == 0:
                return [], []

            prod["product_title"] = prod_details["product_title"]
            prod["product_short_description"] = prod_details["product_short_description"]
            prod["product_details_description"] = prod_details["product_details_description"]
            prod["product_table_details"] = prod_details["product_table_details"]
            prod["product_parameters"] = prod_details["product_parameters"]

            summaries = get_product_summaries_and_tags(
                prod_name=prod_details["product_title"],
                short_desc=prod["product_short_description"],
                long_desc=prod["product_details_description"],
                table=prod["product_table_details"],
                prod_group_desc=prod["product_group_description"],
                prod_family_desc=prod["prod_family_description"],
                category_desc=prod["category_description"]
            )
            prod["product_summary"] = summaries["product_summary"]
            prod["product_summary_hierarchy"] = summaries["hierarchy_summary"]
            prod["product_classifications"] = summaries["classifications"]
            prod["product_param_colour"] = prod_details["product_parameters"]["BARVA"] if "BARVA" in prod_details["product_parameters"] else "NA"
            prod["product_param_exterior"] = prod_details["product_parameters"]["POVRCH"] if "POVRCH" in prod_details["product_parameters"] else "NA"
            prod["product_param_height"] = prod_details["product_parameters"]["VÝŠKA"] if "VÝŠKA" in prod_details["product_parameters"] else "NA"
            prod["product_dimensions"] = summaries["dimensions"]


            # prod["product_summary"] = _get_product_summary(product_title=prod["product_title"], short_desc=prod["product_short_description"], long_desc=prod["product_details_description"], table=prod["product_table_details"])
            # prod["product_summary_hierarchy"] = _get_product_summary_with_hierarchy(
            #     short_desc=prod["product_short_description"],
            #     long_desc=prod["product_details_description"],
            #     table=prod["product_table_details"],
            #     prod_group_desc=prod["product_group_description"],
            #     prod_family_desc=prod["prod_family_description"],
            #     category_desc=prod["category_description"]
            # )

            return [], [prod]
        except Exception as e:
            print(f"   !!!! Failed to extract details for {link}. Skipping. {e}")
            traceback.print_exc()
            return [], []

def main():

    categories_urls = [
        "https://www.best.cz/dlazby",
        "https://www.best.cz/obrubniky",
        "https://www.best.cz/ploty-a-zdi",
        "https://www.best.cz/schodiste-a-palisady",
        "https://www.best.cz/ztracene-bedneni",
        "https://www.best.cz/studny-a-kanalizace",
        "https://www.best.cz/dopravni-infrastruktura",
        "https://www.best.cz/hruba-stavba"
    ]


    all_prod_families: t.List[t.Dict] = []
    for cat_url in tqdm.tqdm(categories_urls, desc="Getting product families in categories", ncols=100):
        webtxt = download_website(url=cat_url)
        if webtxt is None:
            print(f"   !!!! Failed to download category {cat_url}. Skipping.")
            continue
        product_family = extract_links_from_category_page(webpage=webtxt, web_url=cat_url, base_url="https://www.best.cz/")
        all_prod_families.append(product_family)
    # for pf in all_prod_families:
    #     print(f" +++ {pf}")

    all_product_groups: t.List[t.Dict] = []
    for pf in tqdm.tqdm(all_prod_families, desc="Getting product groups in product families", ncols=100):
        for link in pf["links"]:
            webtxt = download_website(link)
            if webtxt is None:
                print(f"   !!!! Failed to download {link} in product family {pf['category_url']}. Skipping.")
                continue
            prod_group = extract_links_from_prod_family_page(webpage=webtxt, web_url=link, base_url="https://www.best.cz/")
            if len(prod_group) == 0:
                continue
            prod_group["category"] = pf["category"]
            prod_group["category_description"] = pf["category_description"]
            prod_group["category_url"] = pf["category_url"]

            all_product_groups.append(prod_group)

    all_product_groups.sort(key=lambda x: x["prod_family_url"], reverse=True)

    all_products: t.List[t.Dict] = []

    for pg in tqdm.tqdm(all_product_groups, desc="Getting product sub-groups in product groups", ncols=100):
        pg_links = pg["prod_family_links"]
        pg_links.sort()
        prods_in_subgroup = []
        for link in pg_links:
            webtxt = download_website(link)
            if webtxt is None:
                print(f"   !!!! Failed to download {link} in product group {pg['prod_family_url']} . Skipping.")
                continue
            if is_product_page(webpage=webtxt):
                product_dict = {
                    "product_url": link,
                    "product_group": pg["prod_family"],
                    "product_group_description": "",
                    "product_group_url": link,
                    "is_product": True
                }
                prods_in_subgroup.append(product_dict)
            else:
                prods_in_subgroup.extend(extract_products_from_product_groups(webpage=webtxt, web_url=link, base_url="https://www.best.cz/"))

        for p_sg in prods_in_subgroup:
            p_sg["category"] = pg["category"]
            p_sg["category_description"] = pg["category_description"]
            p_sg["category_url"] = pg["category_url"]
            p_sg["prod_family"] = pg["prod_family"]
            p_sg["prod_family_description"] = pg["prod_family_description"]
            p_sg["prod_family_url"] = pg["prod_family_url"]
        all_products.extend(prods_in_subgroup)


    # all_products = [{
    #     "product_url": "https://www.best.cz/inbelisima-dreno/antracitova/INBELISIMA8D05",
    #     "is_product": True
    # }]

    finished_products = []
    executor: futures.Executor = futures.ThreadPoolExecutor(max_workers=1000)

    while len(all_products) > 0:
        rexamine_products = []
        products_with_info = []
        all_futures: t.List[futures.Future] = []
        for prod in tqdm.tqdm(all_products, desc="Submitting all products", ncols=100):
            all_futures.append(executor.submit(process_one_product, prod))

        for finished_future in tqdm.tqdm(futures.as_completed(all_futures), total=len(all_futures), ncols=100, desc="Waiting for all product summaries to finish"):
            reex_prods, prods = finished_future.result()
            rexamine_products.extend(reex_prods)
            products_with_info.extend(prods)

        # for prod in tqdm.tqdm(all_products, desc="Getting information about all products", ncols=100):
        #     reex_prods, prods = process_one_product(prod)
        #     rexamine_products.extend(reex_prods)
        #     products_with_info.extend(prods)

        finished_products.extend(products_with_info)
        all_products = rexamine_products

    json.dump(finished_products, open("all_products.json", "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()