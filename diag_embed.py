from sentence_transformers import SentenceTransformer
print("Loading model...")
m = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
print("Model loaded. Encoding...")
emb = m.encode(["hello world"], normalize_embeddings=True)
print("OK. len(vec) =", len(emb[0]))
