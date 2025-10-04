import argparse
import json
from dataclasses import dataclass

from flat_catalog import Collection


@dataclass
class Product:
    name: str
    products: list[str]
    colors: list[str]
    description: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "product_data",
        help="Path to product JSON file (alternative_hierarchy_db.json)",
    )
    parser.add_argument(
        "index_data",
        help="Path to index JSON file (all_products.json)",
    )
    parser.add_argument("--groups", action="store_true", help="Process product groups")
    parser.add_argument("--products", action="store_true", help="Process products")
    parser.add_argument("--colors", action="store_true", help="Process colors")
    args = parser.parse_args()

    with open(args.product_data, "r") as f:
        product_data = json.load(f)
    with open(args.index_data, "r") as f:
        aggr_data = json.load(f)

    if args.groups:
        gs = []
        for group_name, group_dict in product_data["product_group_db"].items():
            assert group_name == group_dict["AH PRODUCT GROUP"]
            prod = Product(
                name=group_dict["AH PRODUCT GROUP"],
                products=group_dict["AH PRODUCT"],
                colors=group_dict["AH PRODUCT ALL COLOURS"],
                description=group_dict["AH PRODUCT GROUP DESCRIPTION"],
            )
            gs.append(prod)
        groups = Collection()
        groups.add_items(
            [p.__dict__ for p in gs],
            [p.description for p in gs],
        )
        groups.save("catalog_db2/groups.npz")

    if args.colors:
        cols = set()
        for group_dict in product_data["product_group_db"].values():
            cols = cols.union(set(group_dict["AH PRODUCT ALL COLOURS"]))
        cols = list(cols)
        colors = Collection()
        colors.add_items(cols, cols)
        colors.save("catalog_db2/colors.npz")

    if args.products:
        prods = set()
        for group_dict in product_data["product_group_db"].values():
            prods = prods.union(set(group_dict["AH PRODUCT"]))
        prods = list(prods)

        url_map = {}
        for item in aggr_data:
            url_map[item["product_title"]] = item["product_url"]

        metadatas = []
        for p in prods:
            metadatas.append({"name": p, "url": url_map[p] if p in url_map else ""})

        products = Collection()
        products.add_items(metadatas, prods)
        products.save("catalog_db2/products.npz")


if __name__ == "__main__":
    main()
