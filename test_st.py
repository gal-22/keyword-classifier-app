from sentence_transformers import SentenceTransformer
try:
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")
    emb = model.encode(["test"])
    print("Encoding success.")
except Exception as e:
    print(f"Error: {e}")
