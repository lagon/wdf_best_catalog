import hashlib
import json
import os
import traceback
import typing as t
import enum

import pydantic as pyd

import openai as oai
import tqdm
from openai import models

import oai_batch
from oai_batch import OAI_Batch


class ExtendCustomerDescriptionSumWorkItem(oai_batch.WorkItem):
    def __init__(self, requested_product: str, additional_data: t.Dict[str, str], model_name: str, work_dir: str):
        super().__init__()
        self._requested_product = requested_product
        self._additional_data = additional_data
        self._model_name = model_name
        self._work_dir = work_dir
        self._response_data: t.Optional[t.Dict[str, str]] = None
        os.makedirs(work_dir, exist_ok=True)

        additional_data_content = []
        for k, v in self._additional_data.items():
            additional_data_content.append(f"{k}={v}")
        additional_data_str = "|||".join(additional_data_content)

        self._item_id = hashlib.md5(";;;".join([requested_product, additional_data_str, model_name]).encode()).hexdigest()

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


    def get_jsonl_list(self) -> t.List[str]:
        extend_description_prompt: str = """Zakazník poptává produkt s následujícím popisem. Nahraď všechny zkratky. Dále zkus odhadnout a rozepsat, 
jak by takový produkt mohl vypadat, identifikuj typ a určení výrobku, požadovaný materiál, 
barvu a rozměry. Napiš jeden kratký odstavec s popisem. Odpovídej v češtině."""
        identify_colour_prompt: str = """Zakazník poptává produkt s následujícím popisem. Pokus se identifikovat z popisu barvu poptavaného zboží.
Pokud barva není zmíněna nebo se nedaří ji určit, odpověz: NENÍ. Vždy odpověz jen jedním slovem."""
        batch_input = [
            self._get_request_body(request_id="extend_description", model_name=self._model_name, prompt=extend_description_prompt, input=self._requested_product),
            self._get_request_body(request_id="colour_id", model_name=self._model_name, prompt=identify_colour_prompt, input=self._requested_product)
        ]

        return batch_input

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"customer-extended-{self.get_id()}.json")

    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        store_object = {}
        store_object["additional_data"] = self._additional_data

        for resp in reponse_jsons:
            custom_id = resp["custom_id"]
            data = resp["response"]["body"]["choices"][0]["message"]["content"]
            store_object[custom_id] = data

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


def _get_category_prompts() -> t.Tuple[str, str]:
    category_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your end task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

First, you have to decide whether the product customer is asking for may fit into one or more of the following categories. 
Below is the list of available categories delimited by hashes #####. Each category has CATEGORY LABEL, it's marketing description 
in DESCRIPTION field and generated SUMMARY field, captuting properties of product families within the category. CATEGORY LABEL, DESCRIPTION and SUMMARY
fields are written in Czech. Categories are:
 
{OPTIONS_DESCRIPTION}

###### END OF CATEGORIES DESCRIPTION
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * product fits into one category (indicated by value 'SINGLE_FOUND_MATCH'),
 * product may fit into one of following categories and all should be explored (indicated by value 'MULTIPLE_MATCHES'), 
 * product does not fit into any category and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').
 
Secondly, output the list of matching categories (indicated by CATEGORY LABEL above) with probability that requested product belongs into given category. 
In all cases output only appropriate categories and if there are none the result is empty list. The structure of the json
must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': '...', 
            'PROBABILITY': 0.95
        }}
    ]
}}
"""
    category_prompt = """
Customer's own description of requested product: '{CUSTOMER_DESCRIPTION}'

Expanded version of the product description as derived by LLM '{EXPANDED_DESCRIPTION}'

What is the best category to match the descriptions?
"""
    return category_instructions, category_prompt

def _get_product_family_prompts() -> t.Tuple[str, str]:
#     prod_fam_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other
# similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
# hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
# materials and services for his building project. Your task is to process a single item from customer's list and going through
# the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
# very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
# significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
# and such items are not in the catalogue nor in the hierarchy.
#
# In the previous step, you have guessed, that the customer's item should belong to:
#  * category (in Czech) '{CATEGORY_LABEL}' with marketing description (in Czech) '{CATEGORY_DESC}' and summary of product families within the category (either in Czech or English) '{CATEGORY_SUM}'.
#
# Now you need to find a product family the product should belong to (if any). Below is the list of available product families
# within '{CATEGORY_LABEL}' category delimited by hashes #####. Each product family has LABEL, it's marketing description in DESCRIPTION field and generated SUMMARY field, captuting properties of product
# groups within the product family. LABEL and DESCRIPTION fields are written in Czech, SUMMARY may be in Czech or English. Product Families are:
#
# {OPTIONS_DESCRIPTION}
#
# ###### END OF PRODUCT FAMILY DESCRIPTIONS
# ##########################################
#
# The output is a JSON. First, the output indicates whether you have found:
#  * product fits into one product family (indicated by value 'SINGLE_FOUND_MATCH'),
#  * product may fit into one of following product families and all should be explored (indicated by value 'MULTIPLE_MATCHES'),
#  * product does not fit into any product family and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').
#
# Secondly, output the list of matching product families with probability that requested product belongs into given product family.
# In all cases output only appropriate product families and if there are none the result is empty list. The structure of the json
# must follow the example below:
# {{
#     'result': 'SINGLE_FOUND_MATCH',
#     'selections': [
#         {{
#             'LABEL': 'Dlazby',
#             'PROBABILITY': 0.95
#         }}
#     ]
# }}"""
    prod_fam_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your end task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

In the previous step, you have guessed, that the customer's item should belong to:
 * category (in Czech) '{CATEGORY_LABEL}' 
    
In the second step you need to find one or more product families the product can belong to, if any. Below is the list of available product families 
delimited by hashes #####. Each product family has PRODUCT FAMILY LABEL, it's marketing description in DESCRIPTION field and generated
SUMMARY field, capturing properties of product groups within the product family. PRODUCT FAMILY LABEL, DESCRIPTION and SUMMARY fields are 
written in Czech. Product Families are:
 
{OPTIONS_DESCRIPTION}

###### END OF PRODUCT FAMILY DESCRIPTIONS
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * product fits into one product family (indicated by value 'SINGLE_FOUND_MATCH'),
 * product may fit into one of following product families and all should be explored (indicated by value 'MULTIPLE_MATCHES'), 
 * product does not fit into any product family and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').

Secondly, output the list of matching product families indicated by corresponding PRODUCT FAMILY LABEL with probability that requested
product belongs into given product family. In all cases output only appropriate product families and if there are none
the result is empty list. The structure of the json must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': '...', 
            'PROBABILITY': 0.95
        }}
    ]
}}"""
    prod_fam_prompt = """
Customer's own description of requested product: '{CUSTOMER_DESCRIPTION}'

Expanded version of the product description as derived by LLM '{EXPANDED_DESCRIPTION}'

What is the best product family to match the descriptions?
"""
    return prod_fam_instructions, prod_fam_prompt

def _get_product_group_prompts() -> t.Tuple[str, str]:
#     product_group_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other
# similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
# hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
# materials and services for his building project. Your task is to process a single item from customer's list and going through
# the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
# very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
# significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
# and such items are not in the catalogue nor in the hierarchy.
#
# In the previous step, you have guessed, that the customer's item should belong to:
#  * category (in Czech) '{CATEGORY_LABEL}' with marketing description (in Czech) '{CATEGORY_DESC}' and summary of product families within the category (either in Czech or English) '{CATEGORY_SUM}'.
#  * product family (in Czech) '{PRODUCT_FAMILY_LABEL}' with description (in Czech) '{PRODUCT_FAMILY_DESC}' and summary of product groups within the product family (either in Czech or English) '{PRODUCT_FAMILY_SUM}'.
#
# Now you need to find a product group the product should belong to (if any). Below is the list of available product groups
# within '{CATEGORY_LABEL}' category and '{PRODUCT_FAMILY_LABEL}' product family delimited by hashes #####. Each product group has LABEL,
# it's marketing description in DESCRIPTION field and generated SUMMARY field, captuting properties of products within the product
# group. LABEL and DESCRIPTION fields are written in Czech, SUMMARY may be in Czech or English. Product groups are:
#
# {OPTIONS_DESCRIPTION}
#
# ###### END OF PRODUCT GROUP DESCRIPTIONS
# ##########################################
#
# The output is a JSON. First, the output indicates whether you have found:
#  * product fits into one product group (indicated by value 'SINGLE_FOUND_MATCH'),
#  * product may fit into one of following product groups and all should be explored (indicated by value 'MULTIPLE_MATCHES'),
#  * product does not fit into any product group and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').
#
# Secondly, output the list of matching product groups with probability that requested product belongs into given product group.
# In all cases output only appropriate product groups and if there are none the result is empty list. The structure of the json
# must follow the example below:
# {{
#     'result': 'SINGLE_FOUND_MATCH',
#     'selections': [
#         {{
#             'LABEL': 'Dlazby',
#             'PROBABILITY': 0.95
#         }}
#     ]
# }}
# """
    product_group_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your end task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

In the previous step, you have guessed, that the customer's item should belong to:
 * category (in Czech) '{CATEGORY_LABEL}' 
 * product family (in Czech) '{PRODUCT_FAMILY_LABEL}' 

Now you need to find one or more product groups the product can belong to, if any. Below is the list of available product groups 
delimited by hashes #####. Each product group has PRODUCT GROUP LABEL, it's marketing description in DESCRIPTION field and generated SUMMARY
field, capturing properties of products within the product group. PRODUCT GROUP LABEL, DESCRIPTION and SUMMARY fields are written in Czech.
Product groups are:
 
{OPTIONS_DESCRIPTION}

###### END OF PRODUCT GROUP DESCRIPTIONS
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * product fits into one product group (indicated by value 'SINGLE_FOUND_MATCH'),
 * product may fit into one of following product groups and all should be explored (indicated by value 'MULTIPLE_MATCHES'), 
 * product does not fit into any product group and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').

Secondly, output the list of matching product groups indicated by PRODUCT GROUP LABEL with probability that requested product belongs into given product group. 
In all cases output only appropriate product groups and if there are none the result is empty list. The structure of the json
must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': '...', 
            'PROBABILITY': 0.95
        }}
    ]
}}
"""
    product_group_prompt = """Customer's own description of requested product: '{CUSTOMER_DESCRIPTION}'

Expanded version of the product description as derived by LLM '{EXPANDED_DESCRIPTION}'

What is the best product group to match the descriptions?
"""

    return product_group_instructions, product_group_prompt

def _get_product_prompts() -> t.Tuple[str, str]:
#     product_selection_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other
# similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
# hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
# materials and services for his building project. Your task is to process a single item from customer's list and going through
# the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
# very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
# significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
# and such items are not in the catalogue nor in the hierarchy.
#
# In the previous step, you have guessed, that the customer's item should belong to:
#  * category (in Czech) '{CATEGORY_LABEL}' with marketing description (in Czech) '{CATEGORY_DESC}' and summary of product families within the category (either in Czech or English) '{CATEGORY_SUM}'.
#  * product family (in Czech) '{PRODUCT_FAMILY_LABEL}' with description (in Czech) '{PRODUCT_FAMILY_DESC}' and summary of product groups within the product family (either in Czech or English) '{PRODUCT_FAMILY_SUM}'.
#  * product group (in Czech) '{PRODUCT_GROUP_LABEL}' with description (in Czech) '{PRODUCT_GROUP_DESC}' and summary of products within the product group (either in Czech or English) '{PRODUCT_GROUP_SUM}'.
#
# Now you need to find a product group the product should belong to (if any). Below is the list of available products
# within '{CATEGORY_LABEL}' category, '{PRODUCT_FAMILY_LABEL}' product family and '{PRODUCT_GROUP_LABEL}' product group
# delimited by hashes #####.
#
# Each product has LABEL, it's web headline description in SHORT-DESCRIPTION field, product details in TECH-DESCRIPTION field,
# SUMMARY field with summary of short and tech description fields. In addition there are few more tags available. Specifically:
# 'COLOUR', 'FINISH', 'DESIGN' and 'AREA OF USE'. If particular value is not available, it's either empty string ('') or NENÍ.
# All the information are written in Czech. Products are:
#
# {OPTIONS_DESCRIPTION}
#
# ###### END OF PRODUCT DESCRIPTIONS
# ##########################################
#
# The output is a JSON. First, the output indicates whether you have found:
#  * a single product matches the description (indicated by value 'SINGLE_FOUND_MATCH'),
#  * multiple products match the description (indicated by value 'MULTIPLE_MATCHES'),
#  * or there is no match what so ever (indicated by value 'NO_MATCH') and we should try different product groups.
# Secondly, output the list of all above products with probability that requested product matches the offered product. The structure of the json must follow the example below:
# {{
#     'result': 'SINGLE_FOUND_MATCH',
#     'selections': [
#         {{
#             'LABEL': 'Dlazby',
#             'PROBABILITY': 0.95
#         }}
#     ]
# }}
# """
    product_selection_instructions = """You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

In the previous step, you have guessed, that the customer's item should belong to:
 * category (in Czech) '{CATEGORY_LABEL}' 
 * product family (in Czech) '{PRODUCT_FAMILY_LABEL}' 
 * product group (in Czech) '{PRODUCT_GROUP_LABEL}' 

Now you need to find one or more products that matches customer's description. Below is the list of available products
delimited by hashes #####.

Each product has PRODUCT LABEL, it's web headline description in SHORT-DESCRIPTION field, product details in TECH-DESCRIPTION field,
SUMMARY field with summary of short and tech description fields. In addition there are few more tags available. Specifically:
'COLOUR', 'FINISH', 'DESIGN' and 'AREA OF USE'. If particular value is not available, it's either empty string ('') or NENÍ.  
All the information are written in Czech. Products are:
 
{OPTIONS_DESCRIPTION}

###### END OF PRODUCT DESCRIPTIONS
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * a single product matches the description (indicated by value 'SINGLE_FOUND_MATCH'),
 * multiple products match the description (indicated by value 'MULTIPLE_MATCHES'), 
 * or there is no match what so ever (indicated by value 'NO_MATCH') and we should try different product groups.
Secondly, output the list of all above products indicated by PRODUCT LABEL with probability that requested product matches the offered product.
The structure of the json must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': '...', 
            'PROBABILITY': 0.95
        }}
    ]
}}
"""
    product_selection_prompt = """
Customer's own description of requested product: '{CUSTOMER_DESCRIPTION}'

Expanded version of the product description as derived by LLM '{EXPANDED_DESCRIPTION}'

What is the best product group to match the descriptions?
"""
    return product_selection_instructions, product_selection_prompt


def _get_prompts(prompt_type: str) -> t.Tuple[str, str]:
    if prompt_type == "CATEGORY":
        return _get_category_prompts()
    elif prompt_type == "PRODUCTFAMILY":
        return _get_product_family_prompts()
    elif prompt_type == "PRODUCTGROUP":
        return _get_product_group_prompts()
    elif prompt_type == "PRODUCT":
        return _get_product_prompts()
    return "", ""



def _get_ids_to_choose_from(selected: str, state: str, product_db: t.Dict[str, t.Any]) -> t.List[str]:
    try:
        if state == "CATEGORY":
            return list(product_db["category_2_prod_family"].keys())
        elif state == "PRODUCTFAMILY":
            return product_db["category_2_prod_family"][selected]
        elif state == "PRODUCTGROUP":
            return product_db["prod_family_2_prod_group"][selected]
        elif state == "PRODUCT":
            return product_db["prod_group_2_product"][selected]
    except Exception as e:
        traceback.print_exception(e)
        print(f"""{e} - for {state} -> {selected} """)

    return []


def _get_selection_texts(option_ids: t.List[str], state: str, product_db: t.Dict[str, t.Any]) -> str:
    option_texts = []
    if state == "CATEGORY":
        for option_id in option_ids:
            lbl_desc = product_db["description_per_category"][option_id]
            lbl_summary = product_db["category_summary"][option_id]
            option_texts.append(f"""########
CATEGORY LABEL: '{option_id}'
DESCRIPTION: '{lbl_desc if lbl_desc != '' else "Není popis, řiď se polem SUMMARY."}'
SUMMARY: '{lbl_summary}'""")

    elif state == "PRODUCTFAMILY":
        for option_id in option_ids:
            lbl_desc = product_db["description_per_product_family"][option_id]
            lbl_summary = product_db["prod_family_summary"][option_id]
            option_texts.append(f"""########
PRODUCT FAMILY LABEL: '{option_id}'
DESCRIPTION: '{lbl_desc if lbl_desc != '' else "Není popis, řiď se polem SUMMARY."}'
SUMMARY: '{lbl_summary}'""")

    elif state == "PRODUCTGROUP":
        for option_id in option_ids:
            lbl_desc = product_db["description_per_product_group"][option_id]
            lbl_summary = product_db["prod_group_summary"][option_id]
            option_texts.append(f"""########
PRODUCT GROUP LABEL: '{option_id}'
DESCRIPTION: '{lbl_desc if lbl_desc != '' else "Není popis, řiď se polem SUMMARY."}'
SUMMARY: '{lbl_summary}'""")
    elif state == "PRODUCT":
        for option_id in option_ids:
            prod_details = product_db["description_per_product"][option_id]
            colour = prod_details["product_parameters"]['BARVA'] if 'BARVA' in prod_details["product_parameters"] else "NENÍ"

            option_texts.append(f"""########
PRODUCT LABEL: {prod_details["title"]}
SHORT-DESCRIPTION: {prod_details["product_short_description"]}
TECH-DESCRIPTION: {prod_details["product_details_description"]} 
SUMMARY: {prod_details["product_summary_hierarchy"]}
COLOUR: '{colour}'
FINISH: '{prod_details["product_classifications"]["finish"] if prod_details["product_classifications"]["finish"] != "" else prod_details["product_parameters"]['POVRCH']}'
DESIGN: {prod_details["product_classifications"]['design']}
AREA OF USE: {prod_details["product_classifications"]['area of use']}
""")

    return "\n".join(option_texts)
            
def _get_upper_hierarchy_texts(past_selected: t.Dict[str, str], state: str, product_db: t.Dict[str, t.Any]) -> t.Dict[str, str]:
    out_data: t.Dict[str, str] = {}
    if past_selected["CATEGORY"] != "":
        out_data["CATEGORY_LABEL"] = past_selected["CATEGORY"]
        out_data["CATEGORY_DESC"] = product_db["description_per_category"][past_selected["CATEGORY"]]
        out_data["CATEGORY_SUM"] = product_db["category_summary"][past_selected["CATEGORY"]]
    if past_selected["PRODUCT FAMILY"] != "":
        out_data["PRODUCT_FAMILY_LABEL"] = past_selected["PRODUCT FAMILY"]
        out_data["PRODUCT_FAMILY_DESC"] = product_db["description_per_product_family"][past_selected["PRODUCT FAMILY"]]
        out_data["PRODUCT_FAMILY_SUM"] = product_db["prod_family_summary"][past_selected["PRODUCT FAMILY"]]
    if past_selected["PRODUCT GROUP"] != "":
        out_data["PRODUCT_GROUP_LABEL"] = past_selected["PRODUCT GROUP"]
        out_data["PRODUCT_GROUP_DESC"] = product_db["description_per_product_group"][past_selected["PRODUCT GROUP"]]
        out_data["PRODUCT_GROUP_SUM"] = product_db["prod_group_summary"][past_selected["PRODUCT GROUP"]]
    return out_data



class ProductResolutionWorkItem(oai_batch.WorkItem):
    def __init__(self, data: t.Dict[str, t.Union[str, t.Dict]], model_name: str, product_db: t.Dict[str, t.Any], work_dir: str):
        self._work_dir = work_dir
        self._data = data
        self._model_name = model_name
        self._product_db = product_db

        data_content = []
        for k, v in self._data.items():
            data_content.append(f"{k}={v}")
        data_str = "|||".join(data_content)

        self._item_id = hashlib.md5(";;;".join([data_str, model_name]).encode()).hexdigest()

    def get_id(self) -> str:
        return self._item_id

    def _get_output_filename(self) -> str:
        return os.path.join(self._work_dir, f"product_selection-{self.get_id()}.json")

    def _get_request_prompt(self) -> t.List[str]:
        return []

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

    def value_or_empty_str(self, dictionary: t.Dict[str, str], key: str) -> str:
        if key in dictionary:
            return dictionary[key]
        else:
            return ""

    def get_jsonl_list(self) -> t.List[str]:
        instruction_str, prompt_str = _get_prompts(prompt_type=self._data["STATE"])
        option_ids_to_choose_from: t.List[str] = _get_ids_to_choose_from(selected=self._data["LAST SELECTED"], state=self._data["STATE"], product_db=self._product_db)
        self._data[f"INSTRUCTIONS {self._data["STATE"]}"] = ""
        self._data[f"PROMPT {self._data["STATE"]}"] = ""

        if len(option_ids_to_choose_from) == 0:
            return []

        if len(option_ids_to_choose_from) == 1:
            instruction_str = f"""Precisely output the user input with nothing added or modified."""
            prompt_str = f"""{{
    "result": "SINGLE_FOUND_MATCH",
    "selections': [
        {{
            "LABEL": "{option_ids_to_choose_from[0]}", 
            "PROBABILITY": 1
        }}
    ]
}}"""
        else:
            selection_texts: str = _get_selection_texts(option_ids=option_ids_to_choose_from, state=self._data["STATE"], product_db=self._product_db)
            instructions_data: t.Dict[str, str] = _get_upper_hierarchy_texts(past_selected={
                "CATEGORY": self.value_or_empty_str(self._data["CATEGORY"], "CATEGORY SELECTION"),
                "PRODUCT FAMILY": self.value_or_empty_str(self._data["PRODUCT FAMILY"], "PRODUCT FAMILY SELECTION"),
                "PRODUCT GROUP": self.value_or_empty_str(self._data["PRODUCT GROUP"], "PRODUCT GROUP SELECTION"),
            }, state=self._data["STATE"], product_db=self._product_db)
            instruction_str = instruction_str.format(OPTIONS_DESCRIPTION=selection_texts, **instructions_data)
            prompt_str = prompt_str.format(CUSTOMER_DESCRIPTION=self._data["CUSTOMER DESC"] ,EXPANDED_DESCRIPTION=self._data["EXTENDED DESC"])

        self._data[f"INSTRUCTIONS {self._data["STATE"]}"] = instruction_str
        self._data[f"PROMPT {self._data["STATE"]}"] = prompt_str

        return [self._get_request_body(request_id="selection", prompt=instruction_str, input=prompt_str, model_name=self._model_name)]


    def is_new(self) -> bool:
        return not os.path.isfile(self._get_output_filename())

    def save_resposes(self, reponse_jsons: t.List[str]) -> None:
        store_object = {}
        store_object["additional_data"] = self._data

        for resp in reponse_jsons:
            custom_id = resp["custom_id"]
            data = resp["response"]["body"]["choices"][0]["message"]["content"]
            data = data.replace("'", "\"") # Make sure all double quotes around (single quotes fail)
            store_object[custom_id] = json.loads(data)

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
            return {
                "additional_data": self._data,
                "selection": {
                    "result": "NO_MATCH",
                    "selections": []
                }
            }
            # raise Exception("Not yet evaluated")

    def save_error_message(self, reponse_jsons: t.List[str]) -> None:
        pass



base_llm_name = "gpt-5-nano"
judge_llm_name = "gpt-5-nano"

class SelectionResult(enum.StrEnum):
    SINGLE_FOUND_MATCH = "SINGLE_FOUND_MATCH"
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
    NO_MATCH = "NO_MATCH"

class SelectionWithProbability(pyd.BaseModel):
    LABEL: str
    PROBABILITY: float

class ListOfSelections(pyd.BaseModel):
    result: SelectionResult
    selections: t.List[SelectionWithProbability]

class LlmJudgeResult(pyd.BaseModel):
    decision: bool
    match_confidence: float
    comments: str

class ResponseType(enum.Enum):
    los = ListOfSelections
    ljr = LlmJudgeResult

def expand_customer_requested_product(request_offer_list: t.List[t.Dict], batch: oai_batch.OAI_Batch, desc_expansion_model_name: str, work_dir: str) -> t.List[t.Dict]:
    desc_expansion_work_items: t.List[ExtendCustomerDescriptionSumWorkItem] = []

    for req_off in request_offer_list:
        additional_data = {
            "FINAL OUTPUT": req_off["OFFERED PRODUCT NAME"],
            "STATE": "EXTEND",
            "CUSTOMER DESC": req_off["CUSTOMER DESCRIPTION"],
            "LAST SELECTED": "NA"
        }
        desc_expansion_work_items.append(
            ExtendCustomerDescriptionSumWorkItem(requested_product=req_off["CUSTOMER DESCRIPTION"], additional_data=additional_data, model_name=desc_expansion_model_name, work_dir=work_dir)
        )


    batch.add_work_items(desc_expansion_work_items)
    batch.run_loop()

    return [wi.get_responses() for wi in desc_expansion_work_items]

def _do_additional_data_structure(last_selected: str, state: str, final_output: str, customer_desc: str, extended_desc: str, req_col: str, category_data: t.Dict, prod_fam_data: t.Dict, prod_grp_data: t.Dict, product_data: t.Dict) -> t.Dict[str, t.Union[str, t.Dict]]:
    return {
            "LAST SELECTED": last_selected,
            "STATE": state,
            "FINAL OUTPUT": final_output,
            "CUSTOMER DESC": customer_desc,
            "EXTENDED DESC": extended_desc,
            "REQ COLOUR": req_col,
            "CATEGORY": category_data,
            "PRODUCT FAMILY": prod_fam_data,
            "PRODUCT GROUP": prod_grp_data,
            "PRODUCT": product_data
        }

def resolve_category(desc_expanded_list: t.List[t.Dict[str, t.Any]], batch: oai_batch.OAI_Batch, product_db: t.Dict[str, t.Any], base_llm_name: str, work_dir: str) -> t.List[t.Dict]:
    category_resolution_work_items: t.List[ProductResolutionWorkItem] = []
    for dewi in desc_expanded_list:
        additional_data = _do_additional_data_structure(
            last_selected="NA",
            state="CATEGORY",
            final_output=dewi["additional_data"]["FINAL OUTPUT"],
            customer_desc=dewi["additional_data"]["CUSTOMER DESC"],
            extended_desc=dewi["extend_description"],
            req_col=dewi["colour_id"],
            category_data={},
            prod_fam_data={},
            prod_grp_data={},
            product_data={}
        )
        category_resolution_work_items.append(ProductResolutionWorkItem(data=additional_data, model_name=base_llm_name, work_dir=work_dir, product_db=product_db))

    batch.add_work_items(category_resolution_work_items)
    batch.run_loop()

    return [wi.get_responses() for wi in category_resolution_work_items]

def _update_final_product_selection(wi_data: t.Dict[str, t.Any], final_product_selection: t.Dict[str, t.List[t.Dict]]):
    title = wi_data["CUSTOMER DESC"]
    if title not in final_product_selection:
        final_product_selection[title] = []
    final_product_selection[title].append(wi_data.copy())


def resolve_product_families(categories_list: t.List[t.Dict[str, t.Any]], final_product_selection: t.Dict[str, t.List[t.Dict]], batch: oai_batch.OAI_Batch, product_db: t.Dict[str, t.Any], base_llm_name: str, work_dir: str) -> t.List[t.Dict[str, t.Any]]:
    prod_family_resolution_work_items: t.List[ProductResolutionWorkItem] = []
    for cat_sel_dta in categories_list:
        if len(cat_sel_dta["selection"]["selections"]) == 0:
            wi_data=_do_additional_data_structure(
                last_selected="",
                state="PRODUCTGROUP",
                final_output=cat_sel_dta["additional_data"]["FINAL OUTPUT"],
                customer_desc=cat_sel_dta["additional_data"]["CUSTOMER DESC"],
                extended_desc=cat_sel_dta["additional_data"]["EXTENDED DESC"],
                req_col=cat_sel_dta["additional_data"]["REQ COLOUR"],
                category_data= {
                    "CATEGORY ALL SELECTIONS": cat_sel_dta["selection"],
                    "CATEGORY SELECTION": "",
                    "CATEGORY SELECTION PROB": 0.0,
                    "CATEGORY INSTRUCTIONS": cat_sel_dta["additional_data"]["INSTRUCTIONS CATEGORY"],
                    "CATEGORY PROMPT": cat_sel_dta["additional_data"]["PROMPT CATEGORY"],
                },
                prod_fam_data={},
                prod_grp_data={},
                product_data={}
            )
            _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)
            continue

        for sel in cat_sel_dta["selection"]["selections"]:
            wi_data = _do_additional_data_structure(
                last_selected=sel["LABEL"],
                state="PRODUCTFAMILY",
                final_output=cat_sel_dta["additional_data"]["FINAL OUTPUT"],
                customer_desc=cat_sel_dta["additional_data"]["CUSTOMER DESC"],
                extended_desc=cat_sel_dta["additional_data"]["EXTENDED DESC"],
                req_col=cat_sel_dta["additional_data"]["REQ COLOUR"],
                category_data= {
                    "CATEGORY ALL SELECTIONS": cat_sel_dta["selection"],
                    "CATEGORY SELECTION": sel["LABEL"],
                    "CATEGORY SELECTION PROB": sel["PROBABILITY"],
                    "CATEGORY INSTRUCTIONS": cat_sel_dta["additional_data"]["INSTRUCTIONS CATEGORY"],
                    "CATEGORY PROMPT": cat_sel_dta["additional_data"]["PROMPT CATEGORY"],
                },
                prod_fam_data={},
                prod_grp_data={},
                product_data={}
            )
            if sel["PROBABILITY"] > 0.01:
                prod_family_resolution_work_items.append(
                    ProductResolutionWorkItem(data=wi_data, model_name=base_llm_name, work_dir=work_dir, product_db=product_db)
                )
            else:
                _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)


    batch.add_work_items(prod_family_resolution_work_items)
    batch.run_loop()

    return [wi.get_responses() for wi in prod_family_resolution_work_items]

def resolve_product_groups(product_families_list: t.List[t.Dict[str, t.Any]], final_product_selection: t.Dict[str, t.List[t.Dict]], batch: oai_batch.OAI_Batch, product_db: t.Dict[str, t.Any], base_llm_name: str, work_dir: str) -> t.List[t.Dict[str, t.Any]]:
    prod_group_resolution_work_items: t.List[ProductResolutionWorkItem] = []
    for pf_sel_dta in product_families_list:
        if len(pf_sel_dta["selection"]["selections"]) == 0:
            wi_data=_do_additional_data_structure(
                last_selected="",
                state="PRODUCTGROUP",
                final_output=pf_sel_dta["additional_data"]["FINAL OUTPUT"],
                customer_desc=pf_sel_dta["additional_data"]["CUSTOMER DESC"],
                extended_desc=pf_sel_dta["additional_data"]["EXTENDED DESC"],
                req_col=pf_sel_dta["additional_data"]["REQ COLOUR"],
                category_data=pf_sel_dta["additional_data"]["CATEGORY"].copy(),
                prod_fam_data={
                    "PRODUCT FAMILY ALL SELECTIONS": pf_sel_dta["selection"],
                    "PRODUCT FAMILY SELECTION": "",
                    "PRODUCT FAMILY SELECTION PROB": 0.0,
                    "PRODUCT FAMILY INSTRUCTIONS": pf_sel_dta["additional_data"]["INSTRUCTIONS PRODUCTFAMILY"],
                    "PRODUCT FAMILY PROMPT": pf_sel_dta["additional_data"]["PROMPT PRODUCTFAMILY"],
                },
                prod_grp_data={},
                product_data={}
            )
            _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)
            continue

        for sel in pf_sel_dta["selection"]["selections"]:
            wi_data=_do_additional_data_structure(
                last_selected=sel["LABEL"],
                state="PRODUCTGROUP",
                final_output=pf_sel_dta["additional_data"]["FINAL OUTPUT"],
                customer_desc=pf_sel_dta["additional_data"]["CUSTOMER DESC"],
                extended_desc=pf_sel_dta["additional_data"]["EXTENDED DESC"],
                req_col=pf_sel_dta["additional_data"]["REQ COLOUR"],
                category_data=pf_sel_dta["additional_data"]["CATEGORY"].copy(),
                prod_fam_data={
                    "PRODUCT FAMILY ALL SELECTIONS": pf_sel_dta["selection"],
                    "PRODUCT FAMILY SELECTION": sel["LABEL"],
                    "PRODUCT FAMILY SELECTION PROB": sel["PROBABILITY"],
                    "PRODUCT FAMILY INSTRUCTIONS": pf_sel_dta["additional_data"]["INSTRUCTIONS PRODUCTFAMILY"],
                    "PRODUCT FAMILY PROMPT": pf_sel_dta["additional_data"]["PROMPT PRODUCTFAMILY"],
                },
                prod_grp_data={},
                product_data={}
            )
            if sel["PROBABILITY"] > 0.01:
                prod_group_resolution_work_items.append(
                    ProductResolutionWorkItem(data=wi_data, model_name=base_llm_name, work_dir=work_dir, product_db=product_db)
                )
            else:
                _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)

    batch.add_work_items(prod_group_resolution_work_items)
    batch.run_loop()

    return [wi.get_responses() for wi in prod_group_resolution_work_items]

def resolve_products(product_groups_list: t.List[t.Dict[str, t.Any]], final_product_selection: t.Dict[str, t.List[t.Dict]], batch: oai_batch.OAI_Batch, product_db: t.Dict[str, t.Any], base_llm_name: str, work_dir: str) -> t.List[t.Dict[str, t.Any]]:
    product_resolution_work_items: t.List[ProductResolutionWorkItem] = []
    for pg_sel_dta in product_groups_list:
        if len(pg_sel_dta["selection"]["selections"]) == 0:
            wi_data = _do_additional_data_structure(
                last_selected="",
                state="PRODUCT",
                final_output=pg_sel_dta["additional_data"]["FINAL OUTPUT"],
                customer_desc=pg_sel_dta["additional_data"]["CUSTOMER DESC"],
                extended_desc=pg_sel_dta["additional_data"]["EXTENDED DESC"],
                req_col=pg_sel_dta["additional_data"]["REQ COLOUR"],
                category_data=pg_sel_dta["additional_data"]["CATEGORY"].copy(),
                prod_fam_data=pg_sel_dta["additional_data"]["PRODUCT FAMILY"].copy(),
                prod_grp_data={
                    "PRODUCT GROUP ALL SELECTIONS": pg_sel_dta["selection"],
                    "PRODUCT GROUP SELECTION": "",
                    "PRODUCT GROUP SELECTION PROB": 0.0,
                    "PRODUCT GROUP INSTRUCTIONS": pg_sel_dta["additional_data"]["INSTRUCTIONS PRODUCTGROUP"],
                    "PRODUCT GROUP PROMPT": pg_sel_dta["additional_data"]["PROMPT PRODUCTGROUP"]
                },
                product_data={}
            )
            _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)
            continue

        for sel in pg_sel_dta["selection"]["selections"]:
            wi_data = _do_additional_data_structure(
                last_selected=sel["LABEL"],
                state="PRODUCT",
                final_output=pg_sel_dta["additional_data"]["FINAL OUTPUT"],
                customer_desc=pg_sel_dta["additional_data"]["CUSTOMER DESC"],
                extended_desc=pg_sel_dta["additional_data"]["EXTENDED DESC"],
                req_col=pg_sel_dta["additional_data"]["REQ COLOUR"],
                category_data=pg_sel_dta["additional_data"]["CATEGORY"].copy(),
                prod_fam_data=pg_sel_dta["additional_data"]["PRODUCT FAMILY"].copy(),
                prod_grp_data={
                    "PRODUCT GROUP ALL SELECTIONS": pg_sel_dta["selection"],
                    "PRODUCT GROUP SELECTION": sel["LABEL"],
                    "PRODUCT GROUP SELECTION PROB": sel["PROBABILITY"],
                    "PRODUCT GROUP INSTRUCTIONS": pg_sel_dta["additional_data"]["INSTRUCTIONS PRODUCTGROUP"],
                    "PRODUCT GROUP PROMPT": pg_sel_dta["additional_data"]["PROMPT PRODUCTGROUP"]
                },
                product_data={}
            )

            if sel["PROBABILITY"] > 0.01:
                product_resolution_work_items.append(
                    ProductResolutionWorkItem(data=wi_data, model_name=base_llm_name, work_dir=work_dir, product_db=product_db)
                )
            else:
                _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)


    batch.add_work_items(product_resolution_work_items)
    batch.run_loop()
    return [wi.get_responses() for wi in product_resolution_work_items]


def main():
    # Create the client
    with open('/home/cepekmir/Sources/wdf_best_catalog/all_product_indices.json', 'rt', encoding='utf-8') as f:
        product_indices = json.load(f)

    request_offer_list = [
    {
        "CUSTOMER DESCRIPTION": "DLAŽBY VEGETAČNÍ Z TVÁRNIC Z PLASTICKÝCH HMOT -  dlaždice tl. 60mm",
        "OFFERED PRODUCT NAME": ""
    },
    {
        "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 150mm",
        "OFFERED PRODUCT NAME": ""
    },
    {
        "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 200mm",
        "OFFERED PRODUCT NAME": ""
    },
    {
        "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV RELIÉF TL 60MM",
        "OFFERED PRODUCT NAME": "BEST - BEATON PRO NEVIDOMÉ ČERVENÁ"
    },
    {
        "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV TL 60MM",
        "OFFERED PRODUCT NAME": "BEST - BEATON ČERVENÁ"
    },
    {
        "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM ŠEDÝCH TL 60MM",
        "OFFERED PRODUCT NAME": "BEST - BEATON PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "NÁSTUPIŠTNÍ OBRUBNÍKY BETONOVÉ - bezbariérový obrubník HK 400/290/1000-P",
        "OFFERED PRODUCT NAME": "BEST - ZASTÁVKOVÝ OBRUBNÍK PŘÍMÝ PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON SVAHOVÝCH TVÁRNIC ŠÍŘ 600MM - KASKÁDOVITÝ SKLUZ",
        "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON TVÁRNIC ŠÍŘ 600MM",
        "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 100MM - výška 250mm",
        "OFFERED PRODUCT NAME": "BEST - SINIA I PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 150mm (nájezdový)",
        "OFFERED PRODUCT NAME": "BEST - MONO NÁJEZDOVÝ ROVNÝ PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 250mm",
        "OFFERED PRODUCT NAME": "BEST - MONO II PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 300mm",
        "OFFERED PRODUCT NAME": "BEST - MONO I PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "ZÁHONOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 50MM - výška 200mm",
        "OFFERED PRODUCT NAME": "BEST - PARKAN II PŘÍRODNÍ"
    },
    {
        "CUSTOMER DESCRIPTION": "ŽLABY A RIGOLY DLÁŽDĚNÉ Z BETONOVÝCH DLAŽDIC - 500x250x80mm",
        "OFFERED PRODUCT NAME": "BEST - NAVIGA PŘÍRODNÍ"
    }
    ]

    all_outputs_list = []
    client: oai.OpenAI = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    work_dir: str = "/home/cepekmir/Sources/wdf_best_catalog/tmp/expand_description/"
    desc_expansion_model_name = "gpt-4.1-mini"
    base_llm_name = "gpt-4.1-mini"
    # desc_expansion_model_name = "gpt-4.1-mini"
    # base_llm_name = "gpt-4.1-mini"
    batch: oai_batch.OAI_Batch = oai_batch.OAI_Batch(client=client, working_dir=work_dir, return_work_items=False, max_work_items_to_add=100, number_active_batches=100)

    final_product_selection: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}

    desc_expansion: t.List[t.Dict] = expand_customer_requested_product(request_offer_list=request_offer_list, batch=batch, desc_expansion_model_name=desc_expansion_model_name, work_dir=work_dir)
    category_selection: t.List[t.Dict] = resolve_category(desc_expanded_list=desc_expansion, batch=batch, product_db=product_indices, base_llm_name=base_llm_name, work_dir=work_dir)
    product_families_selection: t.List[t.Dict[str, t.Any]] = resolve_product_families(categories_list=category_selection, final_product_selection=final_product_selection, batch=batch, product_db=product_indices, base_llm_name=base_llm_name, work_dir=work_dir)
    product_groups_selection: t.List[t.Dict[str, t.Any]] = resolve_product_groups(product_families_list=product_families_selection, final_product_selection=final_product_selection, batch=batch, product_db=product_indices, base_llm_name=base_llm_name, work_dir=work_dir)
    product_selection: t.List[t.Dict[str, t.Any]] = resolve_products(product_groups_list=product_groups_selection, final_product_selection=final_product_selection, batch=batch, product_db=product_indices, base_llm_name=base_llm_name, work_dir=work_dir)

    for prod_sel_dta in product_selection:
        wi_data = _do_additional_data_structure(
            last_selected="",
            state="FINAL PRODUCT SELECTION",
            final_output=prod_sel_dta["additional_data"]["FINAL OUTPUT"],
            customer_desc=prod_sel_dta["additional_data"]["CUSTOMER DESC"],
            extended_desc=prod_sel_dta["additional_data"]["EXTENDED DESC"],
            req_col=prod_sel_dta["additional_data"]["REQ COLOUR"],
            category_data=prod_sel_dta["additional_data"]["CATEGORY"].copy(),
            prod_fam_data=prod_sel_dta["additional_data"]["PRODUCT FAMILY"].copy(),
            prod_grp_data=prod_sel_dta["additional_data"]["PRODUCT GROUP"].copy(),
            product_data= {
                "PRODUCT ALL SELECTIONS": prod_sel_dta["selection"],
                # "PRODUCT SELECTION": sel["LABEL"],
                # "PRODUCT SELECTION PROB": sel["PROBABILITY"],
                "PRODUCT INSTRUCTIONS": prod_sel_dta["additional_data"]["INSTRUCTIONS PRODUCT"],
                "PRODUCT PROMPT": prod_sel_dta["additional_data"]["PROMPT PRODUCT"],
            }
        )
        _update_final_product_selection(wi_data=wi_data, final_product_selection=final_product_selection)

    with open(os.path.join(work_dir, "final_product_selection.json"), "wt", encoding="utf-8") as f:
        json.dump(final_product_selection, f, ensure_ascii=False, indent=4)


    # with open(f"result-{base_llm_name}.txt", "wt", encoding="utf-8") as f:
    #     json.dump(all_outputs_list, f, indent=4, ensure_ascii=False)

#     print(" ---------------------------------------------------------------------------- ")
#     user_input = """DLÁŽDĚNÉ KRYTY Z BETONOVÝCH DLAŽDIC DO LOŽE Z KAMENIVA BET. DLAŽBA TL. 60 MM, TVAR "CIHLA" ČERVENÁ RELIÉFNÍ, LOŽNÁ VRSTVA Z KAMENIVA, FR. 2-4, 4-8 MM, TL. 30 MM
# "digitálně odměřeno ze situace reliéfní dlažba betonová - červená barva: 15,0m2=15,000 [A]m2" "- dodání dlažebního materiálu v požadované kvalitě, dodání materiálu pro předepsané lože
# v tloušťce předepsané dokumentací a pro předepsanou výplň spar - očištění podkladu - uložení dlažby dle předepsaného technologického předpisu včetně předepsané podkladní vrstvy a
# předepsané výplně spar - zřízení vrstvy bez rozlišení šířky, pokládání vrstvy po etapách - úpravu napojení, ukončení podél obrubníků, dilatačních zařízení, odvodňovacích proužků,
# odvodňovačů, vpustí, šachet a pod., nestanoví-li zadávací dokumentace jinak - nezahrnuje postřiky, nátěry - nezahrnuje těsnění podél obrubníků,
# dilatačních zařízení, odvodňovacích proužků, odvodňovačů, vpustí, šachet a pod." """
#     resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)
#
#     print(" ---------------------------------------------------------------------------- ")
#     user_input = """DLÁŽDĚNÉ KRYTY Z BETONOVÝCH DLAŽDIC DO LOŽE Z KAMENIVA BET. DLAŽBA TL. 60 MM, TVAR "CIHLA" ČERVENÁ RELIÉFNÍ PRO NEVIDOMÉ, LOŽNÁ VRSTVA Z KAMENIVA, FR. 2-4, 4-8 MM, TL. 30 MM"""
#     resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)
#
#
#     # print(" ---------------------------------------------------------------------------- ")
#     # user_input = """kabel instalační jádro Cu plné izolace PVC plášť PVC 450/750V (CYKY) 3x4mm2"""
#     # resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)
#
#     print(" ---------------------------------------------------------------------------- ")
#     user_input = """Skruž středová TBV-Q 6d/600 betonová"""
#     resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)
#
#     print(" ---------------------------------------------------------------------------- ")
#     # user_input = """SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - SILNIČNÍ OBRUBA 150/250/1000 MM, VČ. BET. LOŽE C20/25nXF3, TL. MIN. 100 MM digitálně odměřeno ze situace silniční obrubník: 180,0m=180,000 [A]m "Položka zahrnuje: dodání a pokládku betonových obrubníků o rozměrech předepsaných zadávací dokumentací betonové lože i boční betonovou opěrku."""
#     user_input = """SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - SILNIČNÍ OBRUBA 150/250/1000 MM, VČ. BET. LOŽE C20/25nXF3, TL. MIN. 100 MM digitálně odměřeno ze situace silniční obrubník: 180,0m=180,000 [A]m"""
#     resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)
#
# #     print(" ---------------------------------------------------------------------------- ")
# #     user_input = """HLOUBENÍ JAM ZAPAŽ I NEPAŽ TŘ. I
# # VČETNĚ NALOŽENÍ A ODVOZU DO RECYKLAČNÍHO STŘEDISKA, POPLATEK ZA SKLÁDKU UVEDEN V POLOŽCE 014102.a
# # "průměrná hloubka vpusti - 1,5 m
# # odstranění konstrukce vozovoky - 0,56 m
# # výkop v rámci výměny AZ - 0,5 m,
# # plocha výkopu - 1,5 x 1,5 m (2,25 m2)
# # počet navržených vpustí - 5 ks
# # výkop pro UV: 5ks*(2,25m2*0,44m)=4,950 [A]m3"
# # "položka zahrnuje:
# # - vodorovná a svislá doprava, přemístění, přeložení, manipulace s výkopkem
# # - kompletní provedení vykopávky nezapažené i zapažené
# # - ošetření výkopiště po celou dobu práce v něm vč. klimatických opatření
# # - ztížení vykopávek v blízkosti podzemního vedení, konstrukcí a objektů vč. jejich dočasného zajištění
# # - ztížení pod vodou, v okolí výbušnin, ve stísněných prostorech a pod.
# # - příplatek za lepivost
# # - těžení po vrstvách, pásech a po jiných nutných částech (figurách)
# # - čerpání vody vč. čerpacích jímek, potrubí a pohotovostní čerpací soupravy (viz ustanovení k pol. 1151,2)
# # - potřebné snížení hladiny podzemní vody
# # - těžení a rozpojování jednotlivých balvanů
# # - vytahování a nošení výkopku
# # - svahování a přesvah. svahů do konečného tvaru, výměna hornin v podloží a v pláni znehodnocené klimatickými vlivy
# # - ruční vykopávky, odstranění kořenů a napadávek
# # - pažení, vzepření a rozepření vč. přepažování (vyjma štětových stěn)
# # - úpravu, ochranu a očištění dna, základové spáry, stěn a svahů
# # - odvedení nebo obvedení vody v okolí výkopiště a ve výkopišti
# # - třídění výkopku
# # - veškeré pomocné konstrukce umožňující provedení vykopávky (příjezdy, sjezdy, nájezdy, lešení, podpěr. konstr., přemostění, zpevněné plochy, zakrytí a pod.)
# # - nezahrnuje uložení zeminy (na skládku, do násypu) ani poplatky za skládku, vykazují se v položce č.0141**"""
# #     resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)



if __name__ == '__main__':
    main()

