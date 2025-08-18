import json
import os
import typing as t
import enum

import pydantic as pyd

import openai as oai

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


def get_chatbot_response(client: oai.OpenAI, instructions: str, input: str) -> ListOfSelections:
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": input}
        ],
        temperature=0,
        text_format=ListOfSelections

    )
    response = response.output[0].content[0].text
    resp = json.loads(response)
    return resp

def _get_item_descriptions_labels(desc_per_label: t.Dict[str, str]) -> str:
    prompt_part_list = []
    for cat_id, cat_desc in desc_per_label.items():
        prompt_part_list.append(f"""########
LABEL: {cat_id}
DESCRIPTION: {cat_desc}""")
    return "\n".join(prompt_part_list)



# Your task is to read a vague product description as requested by a customer and decide if the requested product belongs into one of the following categories.
# Output a list of categories with probabilities indicating how likely the product belongs into corresponding category. Output a json structure in format:
#     [
#         {{
#             'LABEL': str,
#             'PROBABILITY': float
#         }}
#     ]
#
# Description of categories:
# {_get_item_descriptions_labels(categories)}
#
# Vague product description:


def _get_category_instructions(categories: t.Dict) -> str:
    category_selection_instructions = f"""
You are an expert on products sold by manufacturer of prefabricated concrete slabs, tiles and other items. Your task is to determine which product to offer to a customer.
Customer sends a specification of which product he/she needs for the building project. The specification does not align with the products
you are manufacturing. Your task is to find the best matching product from your catalog. The catalog is organised into a product hierarchy of
categories, product families, product groups and products. Pay attention to details and words in customer's description, as they many
significantly alter the requested product.

First you have to decide whether the customer specified product fits into one of the following categories. The name of the category is indicated by LABEL 
and it's description is written down in DESCRIPTION field. Individual categories are delimited by hashes ##### : 
{_get_item_descriptions_labels(categories)}

###### END OF CATEGORIES DESCRIPTION
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * a single category to which the product belong (indicated by value 'SINGLE_FOUND_MATCH'),
 * multiple categories worth exploring (indicated by value 'MULTIPLE_MATCHES'), 
 * or there is no match what so ever (indicated by value 'NO_MATCH').
Secondly, output the list of matching categories with probability that requested product belongs into given category. In all cases output only appropriate categories and if there are none the result is empty list. The structure of the json must follow the example below:
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
    return category_selection_instructions

def determine_category(customer_description: str, categories: t.Dict, client: oai.OpenAI) -> ListOfSelections:
    category_selection_instructions = _get_category_instructions(categories=categories)
    customer_description = f"""Customer's request: {customer_description}"""

    response: ListOfSelections = get_chatbot_response(client=client, instructions=category_selection_instructions, input=customer_description)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def _get_product_family_instructions(product_families: t.Dict, category_label: str, category_desc: str) -> str:
    prod_family_selection_instructions = f"""
You are an expert on products sold by manufacturer of prefabricated concrete slabs, tiles and other items. Your task is to determine which product to offer to a customer.
Customer sends a specification of which product he/she needs for the building project. The specification does not align with the products
you are manufacturing. Your task is to find the best matching product from your catalog. The catalog is organised into a product hierarchy of
categories, product families, product groups and products. Pay attention to details and words in customer's description, as they many
significantly alter the requested product.

So far, you have determined that customers enquiry belongs to category: '{category_label}' with following description: '{category_desc}'.

Within the category are following product families and the requested product should fit one of them. The name of the product family is indicated by LABEL 
and it's description is written down in DESCRIPTION field. Individual product families are delimited by hashes ##### : 
{_get_item_descriptions_labels(product_families)}

###### END OF PRODUCT FAMILIES DESCRIPTION
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * a single product family to which the product belong (indicated by value 'SINGLE_FOUND_MATCH'),
 * multiple product families worth exploring (indicated by value 'MULTIPLE_MATCHES'), 
 * or there is no match what so ever (indicated by value 'NO_MATCH') and we should try different category.
Secondly, output the list of all product families with probability that requested product belongs into given product family. The structure of the json must follow the example below:
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
    return prod_family_selection_instructions

def determine_product_family(customer_description: str, product_families: t.Dict, category_label: str, category_desc: str, client: oai.OpenAI) -> ListOfSelections:
    prod_fam_selection_instructions = _get_product_family_instructions(product_families=product_families, category_label=category_label, category_desc=category_desc)
    customer_description = f"""Customer's request: {customer_description}"""

    response: ListOfSelections = get_chatbot_response(client=client, instructions=prod_fam_selection_instructions, input=customer_description)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def _get_product_group_instructions(product_groups: t.Dict, category_label: str, category_desc: str, product_family_label: str, product_family_desc: str) -> str:
    product_group_selection_instructions = f"""
You are an expert on products sold by manufacturer of prefabricated concrete slabs, tiles and other items. Your task is to determine which product to offer to a customer.
Customer sends a specification of which product he/she needs for the building project. The specification does not align with the products
you are manufacturing. Your task is to find the best matching product from your catalog. The catalog is organised into a product hierarchy of
categories, product families, product groups and products. Pay attention to details and words in customer's description, as they many
significantly alter the requested product.

So far, you have determined that customers enquiry belongs to product family: '{product_family_label}' with description '{product_family_desc}' and category: '{category_label}' with following description: '{category_desc}'.

Within the product family are following product groups and the requested product should fit one of them. The name of the product group is indicated by LABEL 
and it's description is written down in DESCRIPTION field. Individual product groups are delimited by hashes ##### : 
{_get_item_descriptions_labels(product_groups)}

###### END OF PRODUCT GROUPS DESCRIPTION
##########################################

The output is a JSON. First, the output indicates whether you have found: 
 * a single product group to which the product belong (indicated by value 'SINGLE_FOUND_MATCH'),
 * multiple product groups worth exploring (indicated by value 'MULTIPLE_MATCHES'), 
 * or there is no match what so ever (indicated by value 'NO_MATCH') and we should try different product group.
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
    return product_group_selection_instructions

def determine_product_group(customer_description: str, product_groups: t.Dict[str, str], category_label: str, category_desc: str, product_family_label: str, product_family_desc: str, client: oai.OpenAI) -> ListOfSelections:
    prod_grp_selection_instructions = _get_product_group_instructions(product_groups=product_groups, category_label=category_label, category_desc=category_desc, product_family_label=product_family_label, product_family_desc=product_family_desc)
    customer_description = f"""Customer's request: {customer_description}"""

    response: ListOfSelections = get_chatbot_response(client=client, instructions=prod_grp_selection_instructions, input=customer_description)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def _get_product_instructions(product_groups: t.Dict, category_label: str, category_desc: str, product_family_label: str, product_family_desc: str, product_group_label: str, product_group_desc: str) -> str:
    product_group_selection_instructions = f"""
You are an expert on products sold by manufacturer of prefabricated concrete slabs, tiles and other items. Your task is to determine which product to offer to a customer.
Customer sends a specification of which product he/she needs for the building project. The specification does not align with the products
you are manufacturing. Your task is to find the best matching product from your catalog. The catalog is organised into a product hierarchy of
categories, product families, product groups and products. Pay attention to details and words in customer's description, as they many
significantly alter the requested product.

So far, you have determined that customers enquiry belongs to:
 * product group: '{product_group_label}' with description '{product_group_desc}'
 * product family: '{product_family_label}' with description '{product_family_desc}'
 * category: '{category_label}' with following description: '{category_desc}'

Within the product group are following products and the requested product should fit one of them. The name of the product is indicated by LABEL 
and it's description is written down in DESCRIPTION field. Individual products are delimited by hashes ##### : 
{_get_item_descriptions_labels(product_groups)}

###### END OF PRODUCTS DESCRIPTION
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
    return product_group_selection_instructions

def determine_product(customer_description: str, products: t.Dict[str, str], category_label: str, category_desc: str, product_family_label: str, product_family_desc: str, product_group_label: str, product_group_desc: str, client: oai.OpenAI) -> ListOfSelections:
    prod_selection_instructions = _get_product_instructions(product_groups=products, category_label=category_label, category_desc=category_desc, product_family_label=product_family_label, product_family_desc=product_family_desc, product_group_label=product_group_label, product_group_desc=product_group_desc)
    customer_description = f"""Customer's request: {customer_description}"""

    response: ListOfSelections = get_chatbot_response(client=client, instructions=prod_selection_instructions, input=customer_description)
    response['selections'].sort(key=lambda x: x['PROBABILITY'], reverse=True)
    return response

def _are_selected_label_valid(labels: t.List[str], valid_labels: t.List[str]) -> bool:
    return len(labels) > 0 and all([lbl in valid_labels for lbl in labels])

def _resolve_response_type(selected: ListOfSelections, valid_keys: t.List[str]) -> t.Tuple[SelectionResult, t.List[str]]:
    to_explore = []

    if selected['result'] == SelectionResult.SINGLE_FOUND_MATCH:
        to_explore.append(selected['selections'][0]['LABEL'])
    elif selected['result'] == SelectionResult.MULTIPLE_MATCHES:
        to_explore.extend([item['LABEL'] for item in selected['selections']])
    else:
        to_explore = []
    if not _are_selected_label_valid(to_explore, valid_keys):
        return (SelectionResult.NO_MATCH, [])
    else:
        return (selected['result'], to_explore)

def _resolve_category(product_description: str, product_db: t.Dict[str, t.Any], black_listed_categories: t.List[str], client: oai.OpenAI) -> t.Tuple[SelectionResult, t.List[str]]:
    categories: t.Dict[str, str] = product_db['description_per_category'].copy()
    for bl_cat in black_listed_categories:
        del categories[bl_cat]

    selected_category = determine_category(customer_description=product_description, categories=categories, client=client)
    print(selected_category)
    return _resolve_response_type(selected=selected_category, valid_keys=list(categories.keys()))


def _resolve_product_family(product_description: str, product_db: t.Dict[str, t.Any], black_listed_prod_fams: t.List[str], selected_category_label: str, client: oai.OpenAI) -> t.Tuple[SelectionResult, t.List[str]]:
    selected_category_desc = product_db['description_per_category'][selected_category_label]
    product_families: t.Dict[str, str] = {}
    for prod_fam in product_db['category_2_prod_family'][selected_category_label]:
        product_families[prod_fam] = product_db['description_per_product_family'][prod_fam]
    for bl_prod_fam in black_listed_prod_fams:
        del product_families[bl_prod_fam]

    selected_prod_family = determine_product_family(customer_description=product_description, product_families=product_families, category_label=selected_category_label, category_desc=selected_category_desc, client=client)
    print(selected_prod_family)
    return _resolve_response_type(selected=selected_prod_family, valid_keys=list(product_families.keys()))

def _resolve_product_group(product_description: str, product_db: t.Dict[str, t.Any], black_listed_prod_groups: t.List[str], selected_category_label: str, selected_product_family: str, client: oai.OpenAI) -> t.Tuple[SelectionResult, t.List[str]]:
    selected_category_desc = product_db['description_per_category'][selected_category_label]
    selected_prod_fam_desc = product_db['description_per_product_family'][selected_product_family]

    product_groups: t.Dict[str, str] = {}
    for prod_grp in product_db['prod_family_2_prod_group'][selected_product_family]:
        product_groups[prod_grp] = product_db['description_per_product_group'][prod_grp]
    for bl_prod_fam in black_listed_prod_groups:
        del product_groups[bl_prod_fam]

    selected_prod_group = determine_product_group(customer_description=product_description, product_groups=product_groups, category_label=selected_category_label, category_desc=selected_category_desc, product_family_label=selected_product_family, product_family_desc=selected_prod_fam_desc, client=client)
    print(selected_prod_group)
    return _resolve_response_type(selected=selected_prod_group, valid_keys=list(product_groups.keys()))

def _resolve_final_product(product_description: str, product_db: t.Dict[str, t.Any], black_listed_products: t.List[str], selected_category_label: str, selected_product_family: str, selected_product_group: str, client: oai.OpenAI) -> t.Tuple[SelectionResult, t.List[str]]:
    selected_category_desc = product_db['description_per_category'][selected_category_label]
    selected_prod_fam_desc = product_db['description_per_product_family'][selected_product_family]
    selected_prod_grp_desc = product_db['description_per_product_group'][selected_product_group]

    products: t.Dict[str, str] = {}
    for prod in product_db['prod_group_2_product'][selected_product_group]:
        products[prod] = product_db['description_per_product'][prod]
    for bl_prod in black_listed_products:
        del products[bl_prod]

    selected_prod_group = determine_product(customer_description=product_description, products=products, category_label=selected_category_label, category_desc=selected_category_desc, product_family_label=selected_product_family, product_family_desc=selected_prod_fam_desc, product_group_label=selected_product_group, product_group_desc=selected_prod_grp_desc, client=client)
    print(selected_prod_group)
    return _resolve_response_type(selected=selected_prod_group, valid_keys=list(products.keys()))




def resolve_single_enquired_product(product_description: str, product_db: t.Dict[str, t.Any], client: oai.OpenAI) -> t.Tuple[SelectionResult, t.List[str]]:
    res_type, category = _resolve_category(product_description=product_description, product_db=product_db, black_listed_categories=[], client=client)
    print(f"Best categories: {category}")
    if res_type == SelectionResult.NO_MATCH:
        return SelectionResult.NO_MATCH, []

    product_categories = []
    for sel_cat in category:
        res_type, product_family = _resolve_product_family(product_description=product_description, product_db=product_db, black_listed_prod_fams=[], selected_category_label=sel_cat, client=client)
        print(f"Best product family: {product_family}")
        if res_type == SelectionResult.NO_MATCH:
            return SelectionResult.NO_MATCH, []



    res_type, product_family = _resolve_product_family(product_description=product_description, product_db=product_db, black_listed_prod_fams=[], selected_category_label=sel_category, client=client)
    print(f"Best product family: {product_family}")
    if res_type == SelectionResult.NO_MATCH:
        return SelectionResult.NO_MATCH, []

    # product_family = ["ZÁMKOVÉ A ZAHRADNÍ DLAŽBY"]
    sel_prod_fam = product_family[0]
    res_type, product_group = _resolve_product_group(product_description=product_description, product_db=product_db, black_listed_prod_groups=[], selected_category_label=sel_category, selected_product_family=sel_prod_fam, client=client)
    print(f"Best product group: {product_group}")
    if res_type == SelectionResult.NO_MATCH:
        return SelectionResult.NO_MATCH, []

    # product_group = ["GIGANTICKÁ"]
    sel_prod_group = product_group[0]
    res_type, product = _resolve_final_product(product_description=product_description, product_db=product_db, black_listed_products=[], selected_category_label=sel_category, selected_product_family=sel_prod_fam, selected_product_group=sel_prod_group, client=client)
    print(f"Best product: {product}")

    print(category)
    print(product_family)
    print(product_group)
    print(product)


def main():
    # Create the client
    client = oai.OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
    with open('/home/cepekmir/Sources/wdf_best_catalog/all_product_indices.json', 'rt', encoding='utf-8') as f:
        product_indices = json.load(f)

    print(" ---------------------------------------------------------------------------- ")
    user_input = """DLÁŽDĚNÉ KRYTY Z BETONOVÝCH DLAŽDIC DO LOŽE Z KAMENIVA BET. DLAŽBA TL. 60 MM, TVAR "CIHLA" ČERVENÁ RELIÉFNÍ, LOŽNÁ VRSTVA Z KAMENIVA, FR. 2-4, 4-8 MM, TL. 30 MM 
"digitálně odměřeno ze situace reliéfní dlažba betonová - červená barva: 15,0m2=15,000 [A]m2" "- dodání dlažebního materiálu v požadované kvalitě, dodání materiálu pro předepsané lože 
v tloušťce předepsané dokumentací a pro předepsanou výplň spar - očištění podkladu - uložení dlažby dle předepsaného technologického předpisu včetně předepsané podkladní vrstvy a 
předepsané výplně spar - zřízení vrstvy bez rozlišení šířky, pokládání vrstvy po etapách - úpravu napojení, ukončení podél obrubníků, dilatačních zařízení, odvodňovacích proužků, 
odvodňovačů, vpustí, šachet a pod., nestanoví-li zadávací dokumentace jinak - nezahrnuje postřiky, nátěry - nezahrnuje těsnění podél obrubníků, 
dilatačních zařízení, odvodňovacích proužků, odvodňovačů, vpustí, šachet a pod."
"""
    resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)

    print(" ---------------------------------------------------------------------------- ")
    user_input = """kabel instalační jádro Cu plné izolace PVC plášť PVC 450/750V (CYKY) 3x4mm2"""
    resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)

    print(" ---------------------------------------------------------------------------- ")
    user_input = """Skruž středová TBV-Q 6d/600 betonová"""
    resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)

    print(" ---------------------------------------------------------------------------- ")
    user_input = """SILNIČNÍ A CHODNÍKOVÉ OBRUBY Z BETONOVÝCH OBRUBNÍKŮ ŠÍŘ 150MM
SILNIČNÍ OBRUBA 150/250/1000 MM, VČ. BET. LOŽE C20/25nXF3, TL. MIN. 100 MM
"digitálně odměřeno ze situace 
silniční obrubník: 180,0m=180,000 [A]m"
"Položka zahrnuje: 
dodání a pokládku betonových obrubníků o rozměrech předepsaných zadávací dokumentací 
betonové lože i boční betonovou opěrku."""
    resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)

    print(" ---------------------------------------------------------------------------- ")
    user_input = """HLOUBENÍ JAM ZAPAŽ I NEPAŽ TŘ. I
VČETNĚ NALOŽENÍ A ODVOZU DO RECYKLAČNÍHO STŘEDISKA, POPLATEK ZA SKLÁDKU UVEDEN V POLOŽCE 014102.a
"průměrná hloubka vpusti - 1,5 m 
odstranění konstrukce vozovoky - 0,56 m 
výkop v rámci výměny AZ - 0,5 m, 
plocha výkopu - 1,5 x 1,5 m (2,25 m2) 
počet navržených vpustí - 5 ks 
výkop pro UV: 5ks*(2,25m2*0,44m)=4,950 [A]m3"
"položka zahrnuje: 
- vodorovná a svislá doprava, přemístění, přeložení, manipulace s výkopkem 
- kompletní provedení vykopávky nezapažené i zapažené 
- ošetření výkopiště po celou dobu práce v něm vč. klimatických opatření 
- ztížení vykopávek v blízkosti podzemního vedení, konstrukcí a objektů vč. jejich dočasného zajištění 
- ztížení pod vodou, v okolí výbušnin, ve stísněných prostorech a pod. 
- příplatek za lepivost 
- těžení po vrstvách, pásech a po jiných nutných částech (figurách) 
- čerpání vody vč. čerpacích jímek, potrubí a pohotovostní čerpací soupravy (viz ustanovení k pol. 1151,2) 
- potřebné snížení hladiny podzemní vody 
- těžení a rozpojování jednotlivých balvanů 
- vytahování a nošení výkopku 
- svahování a přesvah. svahů do konečného tvaru, výměna hornin v podloží a v pláni znehodnocené klimatickými vlivy 
- ruční vykopávky, odstranění kořenů a napadávek 
- pažení, vzepření a rozepření vč. přepažování (vyjma štětových stěn) 
- úpravu, ochranu a očištění dna, základové spáry, stěn a svahů 
- odvedení nebo obvedení vody v okolí výkopiště a ve výkopišti 
- třídění výkopku 
- veškeré pomocné konstrukce umožňující provedení vykopávky (příjezdy, sjezdy, nájezdy, lešení, podpěr. konstr., přemostění, zpevněné plochy, zakrytí a pod.) 
- nezahrnuje uložení zeminy (na skládku, do násypu) ani poplatky za skládku, vykazují se v položce č.0141**"""
    resolve_single_enquired_product(product_description=user_input, product_db=product_indices, client=client)



if __name__ == '__main__':
    main()

