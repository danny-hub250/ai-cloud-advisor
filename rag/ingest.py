import os
import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.environ.get("VECTOR_DB_DIR", os.path.join(os.path.dirname(__file__), "..", ".chroma"))

def load_texts():
    docs = []
    for fn in os.listdir(DATA_DIR):
        if fn.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fn), "r", encoding="utf-8") as f:
                docs.append((fn, f.read()))
    return docs

def main():
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_or_create_collection("cloud_advisor")
    docs = load_texts()
    if not docs:
        print("No texts found in rag/data"); return
    ids, documents, metadatas = [], [], []
    for i,(name,text) in enumerate(docs):
        ids.append(f"doc-{i}"); documents.append(text); metadatas.append({"source": name})
    coll.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(documents)} docs into {DB_DIR}")

if __name__ == "__main__":
    main()
