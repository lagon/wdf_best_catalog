import json
import typing as t

if __name__ == '__main__':
    with open('all_products.json') as f:
        product_data = json.load(f)

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

    for pd in product_data:
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

        description_per_product[pd["product_title"]] = {
            "title": pd['product_title'],
            "product_short_description": pd['product_short_description'],
            "product_details_description": pd['product_details_description']
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
        "product_2_prod_group": product_2_prod_group
    }

    with open('all_product_indices.json', 'w', encoding="utf-8") as f:
        json.dump(big_index_thingy, f, ensure_ascii=False, indent=4)

