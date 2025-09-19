import os

import pandas as pd

from catalog import Catalog

DATA_DIR = "data"
CATALOG_DIR = "catalog_db"

catalog = Catalog(CATALOG_DIR)

pd_read_opts = {
    "orient": "records",
    "lines": True,
}

for collection_name, data_file in [
    ("category", os.path.join(DATA_DIR, "category.json")),
    ("prod_family", os.path.join(DATA_DIR, "prod_family.json")),
    ("prod_group", os.path.join(DATA_DIR, "prod_group.json")),
]:
    df = pd.read_json(data_file, **pd_read_opts)
    catalog.upsert_documents(
        collection=collection_name,
        ids=df["name"].tolist(),
        documents=df["text"].tolist(),
        metadatas=df.to_dict(orient="records"),
    )

df = pd.read_json("data/product.json", **pd_read_opts)

chunk_size = 200
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i : i + chunk_size]
    catalog.upsert_documents(
        collection="product",
        ids=[str(i) for i in chunk.index.tolist()],
        documents=chunk["text"].tolist(),
        metadatas=chunk.to_dict(orient="records"),
    )
