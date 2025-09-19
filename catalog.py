import os
from typing import List

import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np


os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]


def local_ef_init():
    return embedding_functions.DefaultEmbeddingFunction()


def openai_ef_init():
    return embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large"
    )


def build_where_clause(field: str, values: List[str] | str | None):
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    return {field: {"$in": values}}


class Catalog:
    def __init__(self, path: str, embedding_function_init=openai_ef_init):
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
        self.client = chromadb.PersistentClient(
            path=path,
            settings=settings,
        )
        self.embedding_function = embedding_function_init()

        configuration = {
            "hnsw": {
                "space": "cosine",
                "ef_construction": 1000,
            }
        }
        self.collections = {
            name: self.client.get_or_create_collection(
                name=name,
                configuration=configuration,  # type: ignore
                embedding_function=None,
            )
            for name in [
                "category",
                "prod_family",
                "prod_group",
                "product",
            ]
        }

    def embed_document(self, document: str) -> np.ndarray:
        return self.embedding_function([document])[0]

    def upsert_documents(
        self,
        collection: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
    ):
        assert len(ids) == len(documents) == len(metadatas)
        self.collections[collection].upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore
            embeddings=self.embedding_function(documents),
        )

    def _query_subcollection(
        self,
        collection: str,
        query_embed: np.ndarray,
        where: dict | None,
    ):
        result = self.collections[collection].query(
            query_embeddings=[query_embed],
            include=["distances", "metadatas"],
            where=where,
            n_results=1000,
        )
        ids = result["ids"][0]
        metas = result["metadatas"][0]  # type: ignore
        dists = result["distances"][0]  # type: ignore
        return ids, metas, dists

    def query_category(self, query_embed: np.ndarray, _: str | List[str] | None = None):
        return self._query_subcollection(
            collection="category",
            query_embed=query_embed,
            where=None,
        )

    def query_prod_family(
        self, query_embed: np.ndarray, category: str | List[str] | None = None
    ):
        return self._query_subcollection(
            collection="prod_family",
            query_embed=query_embed,
            where=build_where_clause("parent", category),
        )

    def query_prod_group(
        self, query_embed: np.ndarray, prod_family: str | List[str] | None = None
    ):
        return self._query_subcollection(
            collection="prod_group",
            query_embed=query_embed,
            where=build_where_clause("parent", prod_family),
        )

    def query_product(
        self, query_embed: np.ndarray, prod_group: str | List[str] | None = None
    ):
        return self._query_subcollection(
            collection="product",
            query_embed=query_embed,
            where=build_where_clause("parent", prod_group),
        )

    def query(self, query_embed: np.ndarray):
        category = self.query_category(query_embed, None)
        prod_family = self.query_prod_family(query_embed, category[0][0])
        prod_group = self.query_prod_group(query_embed, prod_family[0][0])
        product_result = self.query_product(query_embed, prod_group[0][0])
        return {
            "category": {
                "ids": category[0],
                "metadatas": category[1],
                "distances": category[2],
            },
            "prod_family": {
                "ids": prod_family[0],
                "metadatas": prod_family[1],
                "distances": prod_family[2],
            },
            "prod_group": {
                "ids": prod_group[0],
                "metadatas": prod_group[1],
                "distances": prod_group[2],
            },
            "product": {
                "ids": product_result[0],
                "metadatas": product_result[1],
                "distances": product_result[2],
            },
        }
