from pprint import pprint

from catalog import Catalog

catalog = Catalog("catalog_db")
query_text = "jemne betonove dlazby na pokryti plochy u bazenu"
query = catalog.embed_document(query_text)

categories, _, _ = catalog.query_category(query)
prod_families, _, _ = catalog.query_prod_family(query, categories[0])
prod_groups, _, _ = catalog.query_prod_group(query, prod_families[0])
product, metas, dists = catalog.query_product(query, prod_groups[0])
print(f"{categories=}")
print(f"{prod_families=}")
print(f"{prod_groups=}")
print(f"{product=}")
# print(f"{metas=}")

# ---

result = catalog.query(query)
pprint(result)
