import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
# from transformers import pipeline


class RAGTutor:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path).fillna("")

        self.docs = []
        self.meta = []

        for _, row in self.df.iterrows():

            text = f"""
Topic: {row.get('topic', '')}
Concept: {row.get('concept', '')}
Prerequisites: {row.get('prerequisites', '')}
Question: {row.get('question', '')}
Explanation: {row.get('explanation', '')}
"""

            self.docs.append(text)

            self.meta.append({
                "topic": str(row.get("topic", "")),
                "concept": str(row.get("concept", "")),
                "prerequisites": str(row.get("prerequisites", "")),
                "question": str(row.get("question", "")),
                "explanation": str(row.get("explanation", ""))
            })

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings = self.embedder.encode(
            self.docs,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.generator = None

    def retrieve(self, query, top_k=3):

        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, ids = self.index.search(q_emb, top_k)

        results = []

        for i in ids[0]:
            results.append(self.meta[int(i)])

        return results

    def explain_wrong_answer(
        self,
        question,
        selected_answer,
        correct_answer,
        topic,
        concept,
        prerequisites
    ):

        query = f"""
{topic}
{concept}
{question}
{selected_answer}
{prerequisites}
"""

        retrieved = self.retrieve(query, top_k=3)

        best = retrieved[0] if retrieved else {}

        explanation = str(best.get("explanation", "")).strip()
        prereq = str(best.get("prerequisites", prerequisites)).strip()
        concept_name = str(best.get("concept", concept)).strip()

        if not explanation:
            explanation = f"Review the concept {concept_name} carefully."

        # safe retrieval-based RAG fallback
        return f"""
The correct answer is **{correct_answer}**.

Explanation:
{explanation}

Recommended revision:
{prereq if prereq else concept_name}
"""