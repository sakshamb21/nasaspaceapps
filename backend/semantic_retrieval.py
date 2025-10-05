# semantic_retrieval.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class NASAStudySemanticRetriever:
    def __init__(self, csv_path: str, model_name: str = "all-MiniLM-L6-v2", embedding_file: str = "embeddings.npy"):
        # Load dataset
        self.df = pd.read_csv(csv_path).dropna(subset=["Abstract"]).reset_index(drop=True)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Try to load precomputed embeddings, else compute and save
        try:
            self.embeddings = np.load(embedding_file)
            print(f"📂 Loaded precomputed embeddings from {embedding_file}")
        except FileNotFoundError:
            print("🔄 Computing embeddings...")
            self.embeddings = self.model.encode(
                self.df["Abstract"].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            np.save(embedding_file, self.embeddings)
            print(f"✅ Embeddings saved to {embedding_file}")

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (with normalized vectors)
        self.index.add(self.embeddings)
        print(f"✅ FAISS index built with {self.index.ntotal} abstracts")

    def search(self, query: str, top_k: int = 5):
        """
        Perform semantic search over abstracts.
        Returns DataFrame of top_k results with metadata and similarity scores.
        """
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(query_emb), top_k)

        results = self.df.iloc[idxs[0]].copy()
        results["Score"] = scores[0]
        return results[["PMID", "Title", "Year", "Journal", "Abstract", "Link", "Score"]]


if __name__ == "__main__":
    retriever = NASAStudySemanticRetriever("nasa_bioscience_processed1.csv")

    query = "effects of microgravity on bone density"
    results = retriever.search(query, top_k=3)

    print("🔎 Query:", query)
    print("\n--- Top Results ---\n")
    for i, row in results.iterrows():
        print(f"Title: {row['Title']}")
        print(f"Year: {row['Year']} | Journal: {row['Journal']}")
        print(f"Score: {row['Score']:.4f}")
        print(f"Abstract: {row['Abstract'][:300]}...")  # preview
        print(f"Link: {row['Link']}\n")
