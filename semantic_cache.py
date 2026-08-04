import faiss
import numpy as np

from embeddings import embedds

# Get embedding dimension
dimension = len(embedds.embed_query("Hello"))

# Create FAISS index
index = faiss.IndexFlatIP(dimension)

questions = []
answers = []

def add_to_cache(question, answer):
    embedding = embedds.embed_query(question)

    embedding = np.array([embedding], dtype=np.float32)

    index.add(embedding)

    questions.append(question)
    answers.append(answer)

def search_cache(question, threshold=0.90):

    if index.ntotal == 0:
        return None

    embedding = embedds.embed_query(question)
    embedding = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(embedding)

    scores, ids = index.search(embedding, 1)

    score = scores[0][0]
    idx = ids[0][0]
    if idx == -1:
        return None

    if score >= threshold:
        print(f"Semantic Cache Hit: {score:.3f}")
        return answers[idx]

    return None