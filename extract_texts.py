import json
import os

import pandas as pd

DATA_DIR = "data"
INDEX_DATA_FILE = os.path.join(DATA_DIR, "all_product_indices.json")
PRODUCT_DATA_FILE = os.path.join(DATA_DIR, "all_products.json")


def create_df_for_namespace(index_data, namespace, parent):
    index_sumary_key = f"{namespace}_summary"
    summary_dict = index_data[index_sumary_key]

    names = summary_dict.keys()
    texts = [summary_dict[n] for n in names]

    if parent:
        to_parent_key = f"{parent}_2_{namespace}"
        parent_dict = index_data[to_parent_key]
        parents = []
        for n in names:
            ps = [p for p, cs in parent_dict.items() if n in cs]
            parents.append(ps)
    else:
        parents = None

    return pd.DataFrame(
        {
            "name": names,
            "parent": parents,
            "text": texts,
        }
    ).explode("parent")


def create_df_for_products(index_data, product_data):
    index_sumary_key = "description_per_product"
    summary_dict = index_data[index_sumary_key]
    for key, val in summary_dict.items():
        assert key == val["title"]

    records = [p[1] for p in summary_dict.items()]

    data = pd.DataFrame.from_records(records)
    names = data["title"]
    texts = data["product_summary"]

    to_parent_key = "prod_group_2_product"
    parent_dict = index_data[to_parent_key]
    parents = []
    for n in names:
        ps = [p for p, cs in parent_dict.items() if n in cs]
        parents.append(ps)

    url_dict = dict()
    for item in product_data:
        url_dict[item["product_title"]] = item["product_url"]
    urls = [url_dict[n] if n in url_dict else None for n in names]

    return pd.DataFrame(
        {
            "name": names,
            "parent": parents,
            "text": texts,
            "url": urls,
        }
    ).explode("parent")


def main():
    with open(INDEX_DATA_FILE, "r") as f:
        index_data = json.load(f)
    with open(PRODUCT_DATA_FILE, "r") as f:
        product_data = json.load(f)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    pd_write_opts = {
        "orient": "records",
        "force_ascii": False,
        "lines": True,
    }

    for namespace, parent in [
        ("category", None),
        ("prod_family", "category"),
        ("prod_group", "prod_family"),
    ]:
        df = create_df_for_namespace(index_data, namespace, parent)
        df.to_json(f"{DATA_DIR}/{namespace}.json", **pd_write_opts)

    df = create_df_for_products(index_data, product_data)
    df.to_json(f"{DATA_DIR}/product.json", **pd_write_opts)


if __name__ == "__main__":
    main()
