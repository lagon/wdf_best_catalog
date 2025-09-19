import hashlib
import json
import os
import typing as t
import enum

import pydantic as pyd

import openai as oai
import tqdm

base_llm_name = "gpt-5"
judge_llm_name = "gpt-5"

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


def get_chatbot_response(client: oai.OpenAI, instructions: str, input: str, response_type: ResponseType) -> t.Dict[str, t.Any]:
    # print(f"INSTRUCTION: {instructions}")
    # print(f"USER INPUT: {input} ")
    if response_type == ResponseType.los:
        response = client.responses.parse(
            model=base_llm_name,
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input}
            ],
            # temperature=0,
            text_format=ListOfSelections
        )
    elif response_type == ResponseType.ljr:
        response = client.responses.parse(
            model=judge_llm_name,
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input}
            ],
            # temperature=0,
            text_format=LlmJudgeResult
        )

    # response = response.output[0].content[0].text #GPT-4o-mini
    # resp = json.loads(response)
    resp = response.output[1].content[0].parsed
    resp = resp.model_dump()
    return resp

def _get_item_descriptions_labels(description_per_label: t.Dict[str, str], summary_per_label: t.Dict[str, str]) -> str:
    prompt_part_list = []
    for lbl_id in description_per_label.keys():
        lbl_desc = description_per_label[lbl_id]
        lbl_summ = summary_per_label[lbl_id]

        prompt_part_list.append(f"""########
LABEL: {lbl_id}
DESCRIPTION: {lbl_desc if lbl_desc != '' else "Není popis, řiď se polem SUMMARY."}
SUMMARY: {lbl_summ}
""")
    return "\n".join(prompt_part_list)

def _get_product_specific_descriptions_labels(product_info: t.Dict[str, t.Dict[str, str]]) -> str:
    prompt_part_list = []
    for lbl_id in product_info.keys():
        prod_lbl = product_info[lbl_id]["title"]
        prod_short_desc = product_info[lbl_id]["product_short_description"]
        prod_tech_desc = product_info[lbl_id]["product_details_description"]
        prod_summ = product_info[lbl_id]["product_summary"]
        prod_dims = product_info[lbl_id]["product_table_details"]

        prompt_part_list.append(f"""########
LABEL: {prod_lbl}
SHORT-DESCRIPTION: {prod_short_desc if prod_short_desc != '' else "Není popis, řiď se polem SUMMARY."}
TECH-DESCRIPTION: {prod_tech_desc if prod_tech_desc != '' else "Není popis, řiď se polem SUMMARY."}
SUMMARY: {prod_summ}
""")
    return "\n".join(prompt_part_list)



def _get_category_prompts(customer_description: str, category_description: t.Dict[str, str], category_summary: t.Dict[str, str]) -> t.Tuple[str, str]:
    category_instructions = f"""You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

First, you have to decide whether the product customer is asking for fits into one of the following categories. 
Below is the list of available categories delimited by hashes #####. Each category has LABEL, it's marketing description 
in DESCRIPTION field and generated SUMMARY field, captuting properties of product families within the category. LABEL and DESCRIPTION
fields are written in Czech, SUMMARY may be in Czech or English. Categories are:
 
{_get_item_descriptions_labels(description_per_label=category_description, summary_per_label=category_summary)}

###### END OF CATEGORIES DESCRIPTION
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * product fits into one category (indicated by value 'SINGLE_FOUND_MATCH'),
 * product may fit into one of following categories and all should be explored (indicated by value 'MULTIPLE_MATCHES'), 
 * product does not fit into any category and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').
 
Secondly, output the list of matching categories with probability that requested product belongs into given category. 
In all cases output only appropriate categories and if there are none the result is empty list. The structure of the json
must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': 'Dlazby', 
            'PROBABILITY': 0.95
        }}
    ]
}}
"""
    category_prompt = f"""The product the customer is asking about is described as follows: '{customer_description}'. Do any above categories fit the description? If yes, which and how strong is the fit?"""

    return category_instructions, category_prompt

def determine_category(customer_description: str, category_description: t.Dict, category_summary: t.Dict, client: oai.OpenAI) -> t.Dict[str, t.Any]:
    category_instructions, category_prompt = _get_category_prompts(customer_description=customer_description, category_description=category_description, category_summary=category_summary)

    response: t.Dict = get_chatbot_response(client=client, instructions=category_instructions, input=category_prompt, response_type=ResponseType.los)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return {
        "determined": ListOfSelections.model_validate(response),
        "instructions": category_instructions,
        "prompt": category_prompt
    }

def _get_product_family_prompts(customer_description: str, prod_fam_desc: t.Dict[str, str], prod_fam_summ: t.Dict[str, str], category_label: str, category_desc: str, category_summ: str) -> t.Tuple[str, str]:
    pf_selection_instructions = f"""You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

In the previous step, you have guessed, that the customer's item should belong to:
 * category (in Czech) '{category_label}' with marketing description (in Czech) '{category_desc}' and summary of product families within the category (either in Czech or English) '{category_summ}'. 
    
Now you need to find a product family the product should belong to (if any). Below is the list of available product families
within '{category_label}' category delimited by hashes #####. Each product family has LABEL, it's marketing description in DESCRIPTION field and generated SUMMARY field, captuting properties of product
groups within the product family. LABEL and DESCRIPTION fields are written in Czech, SUMMARY may be in Czech or English. Product Families are:
 
{_get_item_descriptions_labels(description_per_label=prod_fam_desc, summary_per_label=prod_fam_summ)}

###### END OF PRODUCT FAMILY DESCRIPTIONS
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * product fits into one product family (indicated by value 'SINGLE_FOUND_MATCH'),
 * product may fit into one of following product families and all should be explored (indicated by value 'MULTIPLE_MATCHES'), 
 * product does not fit into any product family and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').

Secondly, output the list of matching product families with probability that requested product belongs into given product family. 
In all cases output only appropriate product families and if there are none the result is empty list. The structure of the json
must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': 'Dlazby', 
            'PROBABILITY': 0.95
        }}
    ]
}}"""

    pf_selection_prompt = f"""The product the customer is asking about is described as follows: '{customer_description}'. Do any above product families fit the description? If yes, which and how strong is the fit?"""
    return pf_selection_instructions, pf_selection_prompt

def determine_product_family(customer_description: str, prod_fam_desc: t.Dict[str, str], prod_fam_summ: t.Dict[str, str], category_label: str, category_desc: str, category_summ: str, client: oai.OpenAI) -> t.Dict:
    pf_selection_instructions, pf_selection_prompt = _get_product_family_prompts(customer_description=customer_description, prod_fam_desc=prod_fam_desc, prod_fam_summ=prod_fam_summ, category_label=category_label, category_desc=category_desc, category_summ=category_summ)

    response: t.Dict = get_chatbot_response(client=client, instructions=pf_selection_instructions, input=pf_selection_prompt, response_type=ResponseType.los)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return {
        "determined": ListOfSelections.model_validate(response),
        "instructions": pf_selection_instructions,
        "prompt": pf_selection_prompt
    }


def _get_product_group_prompts(customer_description: str, prod_grp_desc: t.Dict[str, str], prod_grp_summ: t.Dict[str, str], category_label: str, category_desc: str, category_summ: str, product_family_label: str, product_family_desc: str, product_family_summ: str) -> t.Tuple[str, str]:
    product_group_instructions = f"""You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

In the previous step, you have guessed, that the customer's item should belong to:
 * category (in Czech) '{category_label}' with marketing description (in Czech) '{category_desc}' and summary of product families within the category (either in Czech or English) '{category_summ}'. 
 * product family (in Czech) '{product_family_label}' with description (in Czech) '{product_family_desc}' and summary of product groups within the product family (either in Czech or English) '{product_family_summ}'. 

Now you need to find a product group the product should belong to (if any). Below is the list of available product groups
within '{category_label}' category and '{product_family_label}' product family delimited by hashes #####. Each product group has LABEL, 
it's marketing description in DESCRIPTION field and generated SUMMARY field, captuting properties of products within the product 
group. LABEL and DESCRIPTION fields are written in Czech, SUMMARY may be in Czech or English. Product groups are:
 
{_get_item_descriptions_labels(description_per_label=prod_grp_desc, summary_per_label=prod_grp_summ)}

###### END OF PRODUCT GROUP DESCRIPTIONS
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * product fits into one product group (indicated by value 'SINGLE_FOUND_MATCH'),
 * product may fit into one of following product groups and all should be explored (indicated by value 'MULTIPLE_MATCHES'), 
 * product does not fit into any product group and is outside of the scope of products in the catalogue (indicated by value 'NO_MATCH').

Secondly, output the list of matching product groups with probability that requested product belongs into given product group. 
In all cases output only appropriate product groups and if there are none the result is empty list. The structure of the json
must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': 'Dlazby', 
            'PROBABILITY': 0.95
        }}
    ]
}}
"""
    product_group_prompt = f"""The product the customer is asking about is described as follows: '{customer_description}'. Do any above product groups fit the description? If yes, which and how strong is the fit?"""

    return product_group_instructions, product_group_prompt


def determine_product_group(customer_description: str, prod_grp_desc: t.Dict[str, str], prod_grp_summ: t.Dict[str, str], category_label: str, category_desc: str, category_summ: str, product_family_label: str, product_family_desc: str, product_family_summ: str, client: oai.OpenAI) -> t.Dict:
    pg_selection_instructions, pg_selection_prompt = _get_product_group_prompts(customer_description=customer_description, prod_grp_desc=prod_grp_desc, prod_grp_summ=prod_grp_summ, category_label=category_label, category_desc=category_desc, category_summ=category_summ, product_family_label=product_family_label, product_family_desc=product_family_desc, product_family_summ=product_family_summ)

    response = get_chatbot_response(client=client, instructions=pg_selection_instructions, input=pg_selection_prompt, response_type=ResponseType.los)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return {
        "determined": ListOfSelections.model_validate(response),
        "instructions": pg_selection_instructions,
        "prompt": pg_selection_prompt
    }


def _get_product_identification_prompts(
        customer_description: str,
        product_info: t.Dict[str, t.Dict[str, str]],
        category_label: str,
        category_desc: str,
        category_summ: str,
        product_family_label: str,
        product_family_desc: str,
        product_family_summ: str,
        product_group_label: str,
        product_group_desc: str,
        product_group_summ: str) -> t.Tuple[str, str]:
    product_selection_instructions = f"""You are selling products sold by manufacturer of prefabricated concrete slabs, tiles and other 
similar products and accessories. All manufactured products are listed in a catalogue. The catalog is organised into a product
hierarchy of categories, product families, product groups and then individual products. A potential customer is looking for
materials and services for his building project. Your task is to process a single item from customer's list and going through
the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy.

In the previous step, you have guessed, that the customer's item should belong to:
 * category (in Czech) '{category_label}' with marketing description (in Czech) '{category_desc}' and summary of product families within the category (either in Czech or English) '{category_summ}'. 
 * product family (in Czech) '{product_family_label}' with description (in Czech) '{product_family_desc}' and summary of product groups within the product family (either in Czech or English) '{product_family_summ}'. 
 * product group (in Czech) '{product_group_label}' with description (in Czech) '{product_group_desc}' and summary of products within the product group (either in Czech or English) '{product_group_summ}'. 

Now you need to find a product group the product should belong to (if any). Below is the list of available products
within '{category_label}' category, '{product_family_label}' product family and '{product_group_label}' product group 
delimited by hashes #####. 
 
 
Each product has LABEL, it's web headline description in SHORT-DESCRIPTION field, product details in TECH-DESCRIPTION field,
SUMMARY field with summary of short and tech description fields. LABEL, SHORT-DESCRIPTION and TECH-DESCRIPTION fields are 
written in Czech, SUMMARY may be in Czech or English. Products are:
 
{_get_product_specific_descriptions_labels(product_info=product_info)}

###### END OF PRODUCT DESCRIPTIONS
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * a single product matches the description (indicated by value 'SINGLE_FOUND_MATCH'),
 * multiple products match the description (indicated by value 'MULTIPLE_MATCHES'), 
 * or there is no match what so ever (indicated by value 'NO_MATCH') and we should try different product groups.
Secondly, output the list of all product families with probability that requested product belongs into given product group. The structure of the json must follow the example below:
{{
    'result': 'SINGLE_FOUND_MATCH',
    'selections': [
        {{
            'LABEL': 'Dlazby', 
            'PROBABILITY': 0.95
        }}
    ]
}}
"""

    product_selection_prompt = f"""The product the customer is asking about is described as follows: '{customer_description}'. Do any above products fit the description? If yes, which and how strong is the fit?"""

    return product_selection_instructions, product_selection_prompt

def determine_product(
        customer_description: str,
        product_info: t.Dict[str, t.Dict[str, str]],
        category_label: str,
        category_desc: str,
        category_summ: str,
        product_family_label: str,
        product_family_desc: str,
        product_family_summ: str,
        product_group_label: str,
        product_group_desc: str,
        product_group_summ: str,
        client: oai.OpenAI) -> t.Dict:

    prod_selection_instructions, product_prompt = _get_product_identification_prompts(
        customer_description=customer_description,
        product_info=product_info,
        category_label=category_label,
        category_desc=category_desc,
        category_summ=category_summ,
        product_family_label=product_family_label,
        product_family_desc=product_family_desc,
        product_family_summ=product_family_summ,
        product_group_label=product_group_label,
        product_group_desc=product_group_desc,
        product_group_summ=product_group_summ)

    response: t.Dict = get_chatbot_response(client=client, instructions=prod_selection_instructions, input=product_prompt, response_type=ResponseType.los)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return {
        "determined": ListOfSelections.model_validate(response),
        "instructions": prod_selection_instructions,
        "prompt": product_prompt
    }

def _are_selected_label_valid(labels: t.List[str], valid_labels: t.List[str]) -> bool:
    return len(labels) > 0 and all([lbl in valid_labels for lbl in labels])

def _resolve_response_type(selected: ListOfSelections, valid_keys: t.List[str]) -> t.Tuple[SelectionResult, t.List[str]]:
    to_explore = []

    if selected.result == SelectionResult.SINGLE_FOUND_MATCH:
        to_explore.append(selected.selections[0].LABEL)
    elif selected.result == SelectionResult.MULTIPLE_MATCHES:
        to_explore.extend([item.LABEL for item in selected.selections])
    else:
        to_explore = []
    if not _are_selected_label_valid(to_explore, valid_keys):
        return (SelectionResult.NO_MATCH, [])
    else:
        return (selected.result, to_explore)

def _resolve_category(product_description: str, product_db: t.Dict[str, t.Any], black_listed_categories: t.List[str], client: oai.OpenAI) -> t.Dict: #t.Tuple[SelectionResult, t.List[str]]:
    category_desc: t.Dict[str, str] = product_db['description_per_category'].copy()
    category_summ: t.Dict[str, str] = product_db['category_summary'].copy()
    for bl_cat in black_listed_categories:
        del category_desc[bl_cat]
        del category_summ[bl_cat]

    selected_category: t.Dict = determine_category(customer_description=product_description, category_description=category_desc, category_summary=category_summ, client=client)
    num_selected, explore_options = _resolve_response_type(selected=selected_category["determined"], valid_keys=list(category_desc.keys()))

    return {
        "num_selected": num_selected,
        "explore_options": explore_options,
        "category_instructions": selected_category["instructions"],
        "category_prompt": selected_category["prompt"]
    }

def _resolve_product_family(product_description: str, product_db: t.Dict[str, t.Any], black_listed_prod_fams: t.List[str], selected_category_label: str, client: oai.OpenAI) -> t.Dict:
    selected_category_desc = product_db['description_per_category'][selected_category_label]
    selected_category_summ = product_db['category_summary'][selected_category_label]

    product_families_desc: t.Dict[str, str] = {}
    product_families_summ: t.Dict[str, str] = {}
    for prod_fam in product_db['category_2_prod_family'][selected_category_label]:
        product_families_desc[prod_fam] = product_db['description_per_product_family'][prod_fam]
        product_families_summ[prod_fam] = product_db['prod_family_summary'][prod_fam]
    for bl_prod_fam in black_listed_prod_fams:
        del product_families_desc[bl_prod_fam]
        del product_families_summ[bl_prod_fam]

    selected_prod_family: t.Dict = determine_product_family(customer_description=product_description, prod_fam_desc=product_families_desc, prod_fam_summ=product_families_summ, category_label=selected_category_label, category_desc=selected_category_desc, category_summ=selected_category_summ, client=client)
    num_selected, explore_options = _resolve_response_type(selected=selected_prod_family["determined"], valid_keys=list(product_families_desc.keys()))
    return {
        "num_selected": num_selected,
        "explore_options": explore_options,
        "prod_fam_instructions": selected_prod_family["instructions"],
        "prod_fam_prompt": selected_prod_family["prompt"]
    }

def _resolve_product_group(product_description: str, product_db: t.Dict[str, t.Any], black_listed_prod_groups: t.List[str], selected_category_label: str, selected_product_family: str, client: oai.OpenAI) -> t.Dict:
    selected_category_desc = product_db['description_per_category'][selected_category_label]
    selected_category_summ = product_db['category_summary'][selected_category_label]

    selected_prod_fam_desc = product_db['description_per_product_family'][selected_product_family]
    selected_prod_fam_summ = product_db['prod_family_summary'][selected_product_family]

    prod_grp_desc: t.Dict[str, str] = {}
    prod_grp_summ: t.Dict[str, str] = {}
    for prod_grp in product_db['prod_family_2_prod_group'][selected_product_family]:
        prod_grp_desc[prod_grp] = product_db['description_per_product_group'][prod_grp]
        prod_grp_summ[prod_grp] = product_db['description_per_product_group'][prod_grp]

    for bl_prod_fam in black_listed_prod_groups:
        del prod_grp_desc[bl_prod_fam]
        del prod_grp_summ[bl_prod_fam]

    selected_prod_group = determine_product_group(
        customer_description=product_description,
        prod_grp_desc=prod_grp_desc,
        prod_grp_summ=prod_grp_summ,
        category_label=selected_category_label,
        category_desc=selected_category_desc,
        category_summ=selected_category_summ,
        product_family_label=selected_product_family,
        product_family_desc=selected_prod_fam_desc,
        product_family_summ=selected_prod_fam_summ,
        client=client)
    # print(selected_prod_group)
    num_selected, explore_options = _resolve_response_type(selected=selected_prod_group["determined"], valid_keys=list(prod_grp_desc.keys()))
    return {
        "num_selected": num_selected,
        "explore_options": explore_options,
        "prod_grp_instructions": selected_prod_group["instructions"],
        "prod_grp_prompt": selected_prod_group["prompt"]
    }

def _resolve_final_product(product_description: str, product_db: t.Dict[str, t.Any], black_listed_products: t.List[str], selected_category_label: str, selected_product_family: str, selected_product_group: str, client: oai.OpenAI) -> t.Dict:
    selected_category_desc = product_db['description_per_category'][selected_category_label]
    selected_category_summ = product_db['category_summary'][selected_category_label]

    selected_prod_fam_desc = product_db['description_per_product_family'][selected_product_family]
    selected_prod_fam_summ = product_db['prod_family_summary'][selected_product_family]

    selected_prod_grp_desc = product_db['description_per_product_group'][selected_product_group]
    selected_prod_grp_summ = product_db['prod_group_summary'][selected_product_group]

    product_info: t.Dict[str, t.Dict[str, str]] = {}
    for prod in product_db['prod_group_2_product'][selected_product_group]:
        product_info[prod] = product_db['description_per_product'][prod]
    for bl_prod in black_listed_products:
        del product_info[bl_prod]

    selected_prods = determine_product(
        customer_description=product_description,
        product_info=product_info,
        category_label=selected_category_label,
        category_desc=selected_category_desc,
        category_summ=selected_category_summ,
        product_family_label=selected_product_family,
        product_family_desc=selected_prod_fam_desc,
        product_family_summ=selected_prod_fam_summ,
        product_group_label=selected_product_group,
        product_group_desc=selected_prod_grp_desc,
        product_group_summ=selected_prod_grp_summ,
        client=client)
    # print(selected_prod_group)
    num_selected, explore_options = _resolve_response_type(selected=selected_prods["determined"], valid_keys=list(product_info.keys()))
    return {
        "num_selected": num_selected,
        "explore_options": explore_options,
        "product_instructions": selected_prods["instructions"],
        "product_prompt": selected_prods["prompt"],
    }

def llm_judge_selected_product(product_description: str, category: str, product_family: str, product_group: str, product: str, product_db: t.Dict, client: oai.OpenAI) -> t.Dict:
    cat_desc = product_db['description_per_category'][category]
    pf_desc = product_db['description_per_product_family'][product_family]
    pg_desc = product_db['description_per_product_group'][product_group]
    prd_desc = product_db['description_per_product'][product]

#
#
#
#      Your task is to process a single item from customer's list and going through
# the hierarchy select the best product. Customer's specification of the product is not aligned with the catalogue and can be
# very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
# significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
# and such items are not in the catalogue nor in the hierarchy.

    llm_judge_instructions = f"""You are selling products sold by manufacturer of prefabricated concrete slabs, 
tiles and other similar products and accessories. All manufactured products are listed in a catalogue. The catalog is 
organised into a product hierarchy of categories, product families, product groups and then individual products.
A potential customer is looking for materials and services for his building project. You are processing a single item
from the list at the time. Previously we have found a proposed match between the product from the catalogue and the item
from the customer's list. Your task is to decide if and how well the selected product from the catalogue matches the requested
item.
Customer's specification of the product is not aligned with the catalogue and can be
very brief, but usually, customer has a specific product in mind. Pay attention to details and words in customer's description, as they many
significantly alter the requested product. Also keep in mind, that customer can ask for items or services that the manufacturer does not sell
and such items are not in the catalogue nor in the hierarchy. 

The proposed product from the catalogue:
 * Category name '{category}', description '{cat_desc}'
 * Product family '{product_family}', description '{pf_desc}'
 * Product group '{product_group}', description '{pg_desc}'
 * Product '{product}', description '{prd_desc}'

Customer's description of the item: '{product_description}'

The output is a JSON object with three fields:
 * decision: a binary flag whether the selected product matches the product description as provided by the customer. 
 * confidence_match: how confident are you that the two products correspond to each other and the customer will accept the suggestion. It's float between 0 and 1. 1 means absolutly confident, 0 means not-at-all. 
 * comments: any comments and reasons you want to give for your decision. 
    """

    llm_judge_prompt=f"""Are the two items aligned?"""

    judgement: t.Dict = get_chatbot_response(client=client, instructions=llm_judge_instructions, input=llm_judge_prompt, response_type=ResponseType.ljr)
    return judgement


def save_detailed_results_from(details_list: t.List, desc_id: str, tag: str) -> None:
    with open(f"details/{desc_id}-{tag}.json", "wt", encoding="utf-8") as f:
        json.dump(details_list, f, ensure_ascii=False, indent=4)


def resolve_single_enquired_product(product_description: str, product_db: t.Dict[str, t.Any], client: oai.OpenAI) -> t.Tuple[SelectionResult, t.List[str]]:
    desc_id: str = hashlib.shake_256(product_description.encode()).hexdigest(4)
    print(f" %%%%%%%%%%%%%%%%% Looking for the best match for {product_description}")
    category_out = _resolve_category(product_description=product_description, product_db=product_db, black_listed_categories=[], client=client)
    print(f"Best categories: {category_out['explore_options']}")
    save_detailed_results_from(details_list=[category_out], desc_id=desc_id, tag="category")
    if category_out['num_selected'] == SelectionResult.NO_MATCH:
        return SelectionResult.NO_MATCH, []

    prod_fams = []
    prod_fams_details = []
    for sel_cat in category_out['explore_options']:
        product_family_out = _resolve_product_family(product_description=product_description, product_db=product_db, black_listed_prod_fams=[], selected_category_label=sel_cat, client=client)
        product_family_out["category"] = sel_cat
        prod_fams_details.append(product_family_out)
        print(f"In category '{sel_cat}' -> best product family: {product_family_out['explore_options']}")
        if product_family_out['num_selected'] != SelectionResult.NO_MATCH:
            for pfam in product_family_out['explore_options']:
                prod_fams.append({
                    "CATEGORY": sel_cat, "PRODUCT_FAMILY": pfam
                })
    save_detailed_results_from(details_list=prod_fams_details, desc_id=desc_id, tag="product_family")

    prod_grps = []
    prod_grp_details = []
    for sel_pf_cat in prod_fams:
        sel_cat = sel_pf_cat["CATEGORY"]
        sel_pf = sel_pf_cat["PRODUCT_FAMILY"]
        prod_grp_out = _resolve_product_group(product_description=product_description, product_db=product_db, black_listed_prod_groups=[], selected_category_label=sel_cat, selected_product_family=sel_pf, client=client)
        prod_grp_out["category"] = sel_cat
        prod_grp_out["product family"] = sel_pf
        prod_grp_details.append(prod_grp_out)
        print(f"In category '{sel_cat}' & product family '{sel_pf}' -> Best product group: {prod_grp_out['explore_options']}")
        if prod_grp_out["num_selected"] != SelectionResult.NO_MATCH:
            for pgrp in prod_grp_out['explore_options']:
                prod_grps.append({
                    "CATEGORY": sel_cat, "PRODUCT_FAMILY": sel_pf, "PRODUCT_GROUP": pgrp
                })
    save_detailed_results_from(details_list=prod_grp_details, desc_id=desc_id, tag="product_family")

    prods = []
    prods_out = []
    for sel_pg_pf_cat in prod_grps:
        sel_cat = sel_pg_pf_cat["CATEGORY"]
        sel_pf = sel_pg_pf_cat["PRODUCT_FAMILY"]
        sel_pg = sel_pg_pf_cat["PRODUCT_GROUP"]

        product_out = _resolve_final_product(product_description=product_description, product_db=product_db, black_listed_products=[], selected_category_label=sel_cat, selected_product_family=sel_pf, selected_product_group=sel_pg, client=client)
        product_out["category"] = sel_cat
        product_out["product family"] = sel_pf
        product_out["product group"] = sel_pg
        prods_out.append(product_out)
        print(f"In category '{sel_cat}' & product family '{sel_pf}' & product group: {sel_pg} --> Best product: {product_out['explore_options']}")
        if product_out['num_selected'] != SelectionResult.NO_MATCH:
            for prd in product_out['explore_options']:
                prods.append({
                    "CATEGORY": sel_cat, "PRODUCT_FAMILY": sel_pf, "PRODUCT_GROUP": sel_pg, "PRODUCT": prd
                })

    save_detailed_results_from(details_list=prods_out, desc_id=desc_id, tag="product_group")
    # prods = [{'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II PŮLKA PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO I PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - RONDA PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'RONDA'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - RONDA POLOMĚR 1 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'RONDA'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - RONDA POLOMĚR 0,5 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'RONDA'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - RONDA POLOMĚR 16 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'RONDA'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - KERBO PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'KERBO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - KERBO POLOMĚR 1 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'KERBO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - KERBO POLOMĚR 0,5 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'KERBO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - KERBO POLOMĚR 16 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'KERBO'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II POLOMĚR 1 VNITŘNÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO DOPLŇKY'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II POLOMĚR 1 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO DOPLŇKY'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II POLOMĚR 0,5 VNITŘNÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO DOPLŇKY'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II POLOMĚR 0,5 VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO DOPLŇKY'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II ROHOVÝ VNĚJŠÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO DOPLŇKY'}, {'CATEGORY': 'OBRUBNÍKY', 'PRODUCT': 'BEST - MONO II ROHOVÝ VNITŘNÍ PŘÍRODNÍ', 'PRODUCT_FAMILY': 'SILNIČNÍ A ZASTÁVKOVÉ OBRUBNÍKY', 'PRODUCT_GROUP': 'MONO DOPLŇKY'}]

    print()
    all_judgements = []
    for select_prod in tqdm.tqdm(prods):
        sel_cat = select_prod["CATEGORY"]
        sel_pf  = select_prod["PRODUCT_FAMILY"]
        sel_pg  = select_prod["PRODUCT_GROUP"]
        sel_prd = select_prod["PRODUCT"]


        judg = llm_judge_selected_product(product_description=product_description, category=sel_cat, product_family=sel_pf, product_group=sel_pg, product=sel_prd, product_db=product_db, client=client)
        judg["CATEGORY"] = select_prod['CATEGORY']
        judg["PRODUCT_FAMILY"] = select_prod['PRODUCT_FAMILY']
        judg["PRODUCT_GROUP"] = select_prod['PRODUCT_GROUP']
        judg["PRODUCT"] = select_prod['PRODUCT']
        all_judgements.append(judg)
    all_judgements.sort(key=lambda x: x['decision'])
    all_judgements.sort(key=lambda x: x["match_confidence"])

    save_detailed_results_from(details_list=all_judgements, desc_id=desc_id, tag="judgement")
    for judg in all_judgements:
        print(f" >> {judg['CATEGORY']} >> {judg['PRODUCT_FAMILY']} >> {judg['PRODUCT_GROUP']} >> {judg['PRODUCT']} -- {judg['match_confidence']} ++ {judg['comments']}")

    return all_judgements


def main():
    # Create the client
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    with open('/home/cepekmir/Sources/wdf_best_catalog/all_product_indices.json', 'rt', encoding='utf-8') as f:
        product_indices = json.load(f)

    request_offer_list = [
    # {
    #     "CUSTOMER DESCRIPTION": "DLAŽBY VEGETAČNÍ Z TVÁRNIC Z PLASTICKÝCH HMOT -  dlaždice tl. 60mm",
    #     "OFFERED PRODUCT NAME": ""
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 150mm",
    #     "OFFERED PRODUCT NAME": ""
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "DRENÁŽNÍ VÝUSŤ Z BETON DÍLCŮ - pro drenáž DN 200mm",
    #     "OFFERED PRODUCT NAME": ""
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV RELIÉF TL 60MM",
    #     "OFFERED PRODUCT NAME": "BEST - BEATON PRO NEVIDOMÉ ČERVENÁ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM BAREV TL 60MM",
    #     "OFFERED PRODUCT NAME": "BEST - BEATON ČERVENÁ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "KRYTY Z BETON DLAŽDIC SE ZÁMKEM ŠEDÝCH TL 60MM",
    #     "OFFERED PRODUCT NAME": "BEST - BEATON PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "NÁSTUPIŠTNÍ OBRUBNÍKY BETONOVÉ - bezbariérový obrubník HK 400/290/1000-P",
    #     "OFFERED PRODUCT NAME": "BEST - ZASTÁVKOVÝ OBRUBNÍK PŘÍMÝ PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON SVAHOVÝCH TVÁRNIC ŠÍŘ 600MM - KASKÁDOVITÝ SKLUZ",
    #     "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
    # },
    {
        "CUSTOMER DESCRIPTION": "PŘÍKOPOVÉ ŽLABY Z BETON TVÁRNIC ŠÍŘ 600MM",
        "OFFERED PRODUCT NAME": "BEST - ŽLAB I PŘÍRODNÍ"
    }
    # {
    #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 100MM - výška 250mm",
    #     "OFFERED PRODUCT NAME": "BEST - SINIA I PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 150mm (nájezdový)",
    #     "OFFERED PRODUCT NAME": "BEST - MONO NÁJEZDOVÝ ROVNÝ PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 250mm",
    #     "OFFERED PRODUCT NAME": "BEST - MONO II PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM - výška 300mm",
    #     "OFFERED PRODUCT NAME": "BEST - MONO I PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "ZÁHONOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 50MM - výška 200mm",
    #     "OFFERED PRODUCT NAME": "BEST - PARKAN II PŘÍRODNÍ"
    # },
    # {
    #     "CUSTOMER DESCRIPTION": "ŽLABY A RIGOLY DLÁŽDĚNÉ Z BETONOVÝCH DLAŽDIC - 500x250x80mm",
    #     "OFFERED PRODUCT NAME": "BEST - NAVIGA PŘÍRODNÍ"
    # }
    ]

    all_outputs_list = []
    for req_off in request_offer_list:
        all_outputs_list.append({
            "Product": req_off,
            "Decisions": resolve_single_enquired_product(product_description=req_off["CUSTOMER DESCRIPTION"], product_db=product_indices, client=client)
        })

    with open(f"result-{base_llm_name}.txt", "wt", encoding="utf-8") as f:
        json.dump(all_outputs_list, f, indent=4, ensure_ascii=False)

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

