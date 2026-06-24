# sanity_check.py
import sys
print("Python:", sys.executable)
try:
    import chromadb
    print("ChromaDB version:", chromadb.__version__)
except Exception as e:
    print("ChromaDB import failed:", e)
