import json
import os
import typing as t

import pydantic as pyd

import openai as oai

class SelectionWithProbability(pyd.BaseModel):
    LABEL: str
    PROBABILITY: float

class ListOfSelections(pyd.BaseModel):
    selections: t.List[SelectionWithProbability]


def get_chatbot_response(client: oai.OpenAI, instructions: str, input: str) -> t.List:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%% INSTRUCTIONS")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(instructions)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%% USER INPUT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(input)

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": input}
        ],
        # text={"format": {"type": "json_object"}}
        text_format=ListOfSelections

    )
    response = response.output[0].content[0].text
    print(response)
    resp = json.loads(response)
    return resp["selections"]


def determine_category(customer_description: str, categories: t.Dict, client: oai.OpenAI) -> t.List:
    category_prompt_part_list = []
    for cat_id, cat_desc in categories.items():
        category_prompt_part_list.append(f"""
########
CATEGORY LABEL: {cat_id}
CATEGORY DESCRIPTION: {cat_desc}
""")

#     category_selection_instructions = f"""
# Your task is to read a vague product description as requested by a customer and decide if the requested product belongs into one of the following categories.
# If the product does not belong into any category, just say NONE. Otherwise output a CATEGORY LABEL of the corresponding category. Respond with just the label and nothing more.
#
# {('\n'.join(category_prompt_part_list))}
#
# Vague product description:
# """
    category_selection_instructions = f"""
Your task is to read a vague product description as requested by a customer and decide if the requested product belongs into one of the following categories.
Output a list of categories with probabilities indicating how likely the product belongs into corresponding category. Output a json structure in format:
[
    {{
        'LABEL': str, 
        'PROBABILITY': float
    }}
]

Description of categories:
{('\n'.join(category_prompt_part_list))}

Vague product description:
"""

    response = get_chatbot_response(client=client, instructions=category_selection_instructions, input=customer_description)
    response.sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def determine_product_family(customer_description: str, product_families: t.Dict, client: oai.OpenAI) -> t.List:
    prod_family_prompt_part_list = []
    for cat_id, cat_desc in product_families.items():
        prod_family_prompt_part_list.append(f"""
########
PRODUCT FAMILY LABEL: {cat_id}
PRODUCT FAMILY DESCRIPTION: {cat_desc}
""")

    category_selection_instructions = f"""
Your task is to read a vague product description as requested by a customer and decide if the requested product belongs into one of the following product families.
Output a list of product families with probabilities indicating how likely the product belongs into corresponding product family. Output a json structure in format:
 [
    {{
        'LABEL': str, 
        'PROBABILITY': float
    }}
]  

Description of product families:
{('\n'.join(prod_family_prompt_part_list))}

Vague product description:
"""
    response = get_chatbot_response(client=client, instructions=category_selection_instructions, input=customer_description)
    response.sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def determine_product_group_probabilities(customer_description: str, product_groups: t.Dict[str, str], client: oai.OpenAI) -> t.List:
    prod_group_prompt_part_list = []
    for cat_id, cat_desc in product_groups.items():
        prod_group_prompt_part_list.append(f"""
    ########
    LABEL: {cat_id}
    DESCRIPTION: {cat_desc}
    """)

    category_selection_instructions = f"""
    Your task is to read a vague product description as requested by a customer and decide if the requested product belongs into one of the following product groups.
    Output a list of product group with probabilities indicating how likely the product belongs into corresponding product group. Output a json structure in format:
     [
        {{
            'LABEL': str, 
            'PROBABILITY': float
        }}
    ]  
    
    Description of product groups:
    {('\n'.join(prod_group_prompt_part_list))}
    
    Vague product description:
    """
    response = get_chatbot_response(client=client, instructions=category_selection_instructions, input=customer_description)
    response.sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def determine_product_probabilities(customer_description: str, products: t.Dict[str, str], client: oai.OpenAI) -> t.List:
    prod_group_prompt_part_list = []
    for cat_id, cat_desc in products.items():
        prod_group_prompt_part_list.append(f"""
    ########
    LABEL: {cat_id}
    DESCRIPTION: {cat_desc}
    """)

    category_selection_instructions = f"""
    Your task is to read a vague product description as requested by a customer and decide if the requested product is one of the following products.
    Output a list of products with probabilities indicating how likely the requested product corresponds to described product. Output a json structure in format:
     [
        {{
            'LABEL': str, 
            'PROBABILITY': float
        }}
    ]  
    
    Description of individual products:
    {('\n'.join(prod_group_prompt_part_list))}
    
    Vague product description:
    """
    response = get_chatbot_response(client=client, instructions=category_selection_instructions, input=customer_description)
    response.sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response


def main():
    # Create the client
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    with open('/home/cepekmir/Sources/wdf_best_catalog/all_product_indices.json', 'rt', encoding='utf-8') as f:
        product_indices = json.load(f)

    categories = product_indices['description_per_category']

    user_input = """
DLÁŽDĚNÉ KRYTY Z BETONOVÝCH DLAŽDIC DO LOŽE Z KAMENIVA BET. DLAŽBA TL. 60 MM, TVAR "CIHLA" ČERVENÁ RELIÉFNÍ, LOŽNÁ VRSTVA Z KAMENIVA, FR. 2-4, 4-8 MM, TL. 30 MM "digitálně odměřeno ze situace reliéfní dlažba betonová - červená barva: 15,0m2=15,000 [A]m2" "- dodání dlažebního materiálu v požadované kvalitě, dodání materiálu pro předepsané lože v tloušťce předepsané dokumentací a pro předepsanou výplň spar - očištění podkladu - uložení dlažby dle předepsaného technologického předpisu včetně předepsané podkladní vrstvy a předepsané výplně spar - zřízení vrstvy bez rozlišení šířky, pokládání vrstvy po etapách - úpravu napojení, ukončení podél obrubníků, dilatačních zařízení, odvodňovacích proužků, odvodňovačů, vpustí, šachet a pod., nestanoví-li zadávací dokumentace jinak - nezahrnuje postřiky, nátěry - nezahrnuje těsnění podél obrubníků, dilatačních zařízení, odvodňovacích proužků, odvodňovačů, vpustí, šachet a pod."
"""
    category_propabilities = determine_category(customer_description=user_input, categories=categories, client=client)
    print(category_propabilities)
    determined_category = category_propabilities[0]['LABEL']
    # determined_category = 'DLAŽBY'

    if determined_category not in categories.keys():
        print("WRONG CATEGORY")
        return

    determined_category_desc = product_indices['description_per_category'][determined_category]
    product_family_names = product_indices['category_2_prod_family'][determined_category]
    product_families = {}
    for prod_fam in product_family_names:
        product_families[prod_fam] = product_indices['description_per_product_family'][prod_fam]

    product_family_probabilities = determine_product_family(customer_description=user_input, product_families=product_families, client=client)
    print(product_family_probabilities)
    determined_product_family = product_family_probabilities[0]['LABEL']
    # determined_product_family = "ZÁMKOVÉ A ZAHRADNÍ DLAŽBY"
    print(determined_product_family)

    determined_product_family_desc = product_indices['description_per_product_family'][determined_product_family]
    product_group_names = product_indices['prod_family_2_prod_group'][determined_product_family]
    product_groups = {}
    for prod_grp in product_group_names:
        product_groups[prod_grp] = product_indices['description_per_product_group'][prod_grp]

    if len(product_groups) > 1:
        product_group_probabilities = determine_product_group_probabilities(customer_description=user_input, product_groups=product_groups, client=client)
        print(product_group_probabilities)
        determined_product_group = product_group_probabilities[0]['LABEL']
    else:
        determined_product_group = list(product_groups.keys())[0]
    print(determined_product_group)


    determined_product_group_desc = product_indices['description_per_product_group'][determined_product_group]
    product_names = product_indices['prod_group_2_product'][determined_product_group]
    products = {}
    for prods in product_names:
        products[prods] = product_indices['description_per_product'][prods]

    product_probabilities = determine_product_probabilities(customer_description=user_input, products=products, client=client)
    print(product_probabilities)
    determined_product = product_probabilities[0]['LABEL']
    print(determined_product)






if __name__ == '__main__':
    main()

