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
    parser.add_argument("product_data", help="Path to product JSON file")
    args = parser.parse_args()

    with open(args.product_data, "r") as f:
        product_data = json.load(f)

    products = []
    for group_name, group_dict in product_data["product_group_db"].items():
        assert group_name == group_dict["AH PRODUCT GROUP"]
        prod = Product(
            name=group_dict["AH PRODUCT GROUP"],
            products=group_dict["AH PRODUCT"],
            colors=group_dict["AH PRODUCT ALL COLOURS"],
            description=group_dict["AH PRODUCT GROUP DESCRIPTION"],
        )
        products.append(prod)

    products = products[:3]  # TODO: Limit for testing

    catalog = Collection()
    catalog.add_items(products, [p.description for p in products])
    catalog.save("flat_catalog.npz")


if __name__ == "__main__":
    main()
