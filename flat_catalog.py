import os

import numpy as np
from openai import OpenAI
from pybase64 import b64decode


class OpenAIEmbedder:
    embed_size = 3072

    def __init__(self):
        assert os.environ.get("OPENAI_API_KEY") is not None
        self.client = OpenAI()

    def __call__(self, texts: str | list[str]):
        if isinstance(texts, str):
            return self._embed([texts])[0]
        return self._embed(texts)

    def _embed(self, texts: list[str]):
        response = self.client.embeddings.create(
            input=[t for t in texts],
            model="text-embedding-3-large",
            encoding_format="base64",
        )
        embeds_b64 = [r.embedding for r in response.data]
        embeds_bytes = [b64decode(e) for e in embeds_b64]  # type: ignore
        embeds_floats = [np.frombuffer(e, dtype=np.float32) for e in embeds_bytes]
        embeddings = np.vstack(embeds_floats)
        return embeddings


class Collection:
    embed_size = 3072

    def __init__(self, file_path: str | None = None):
        self.items = []
        self.embedder = OpenAIEmbedder()
        self.embeddings = np.empty(
            (0, Collection.embed_size),
            dtype=np.float32,
        )
        if file_path:
            self.load(file_path)

    def load(self, file_path: str):
        data = np.load(file_path, allow_pickle=True)
        self.items = data["items"]
        self.embeddings = data["embeddings"]

    def save(self, file_path: str):
        np.savez_compressed(
            file=file_path,
            allow_pickle=True,
            items=self.items,  # type: ignore
            embeddings=self.embeddings,
        )

    def add_items(self, items: list, texts: list[str]):
        embeddings = self.embedder(texts)
        return self._add(items, embeddings)

    def _add(self, items: list, embeddings: np.ndarray):
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == Collection.embed_size
        assert len(items) == embeddings.shape[0]

        # normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        start, end = len(self.items), len(self.items) + len(items)
        self.items.extend(items)
        self.embeddings = np.vstack((self.embeddings, embeddings))
        return np.arange(start, end)

    def search_items(self, query_text: str, top_k: int | None = None):
        query_embedding = self.embedder(query_text)
        return self._search(query_embedding, top_k)

    def _search(self, query_embedding: np.ndarray, top_k: int | None = None):
        assert query_embedding.ndim == 1
        assert query_embedding.shape[0] == Collection.embed_size

        if top_k is None:
            top_k = len(self.items)

        # cosine similarity
        dists = self.embeddings.dot(query_embedding)
        indices = np.argsort(dists)[::-1]
        top_k_indices = indices[:top_k]
        # result = [(dists[i], self.items[i]) for i in top_k_indices]
        result = [self.items[i] for i in top_k_indices], dists[top_k_indices]
        return result
