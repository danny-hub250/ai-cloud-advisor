import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_DIR = os.environ.get("VECTOR_DB_DIR", os.path.join(os.path.dirname(__file__), "..", ".chroma"))

def get_collection():
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection("cloud_advisor", embedding_function=ef)

def rag_search(query: str, k: int = 4):
    coll = get_collection()
    res = coll.query(query_texts=[query], n_results=k)
    results = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        class R:
            page_content = doc
            metadata = meta
            score = 1.0 - float(dist) if dist is not None else 0.0
        results.append(R())
    return results
