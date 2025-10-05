"""
kg_builder.py

Build a knowledge graph from a CSV of papers and a .npy file of precomputed embeddings.

Requirements:
- Python 3.8+
- numpy, pandas, scikit-learn, networkx

Primary classes/functions:
- KnowledgeGraphBuilder: main class to create and query graph
- load_csv_embeddings: helper loader
- export_graph_json: export for frontend visualizations


"""

from typing import List, Dict, Optional, Tuple, Any
import logging
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# Configure logger
logger = logging.getLogger("kg_builder")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(ch)


def load_csv_embeddings(csv_path: str, embeddings_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the CSV with columns: Title, Abstract, PCMID, Link, Summary
    and the numpy embeddings file (.npy) corresponding to the same row order.

    Args:
        csv_path: path to CSV file
        embeddings_path: path to .npy embeddings file (shape: [n_papers, dim])

    Returns:
        df: pandas DataFrame (rows correspond to embeddings rows)
        embeddings: numpy array of floats
    """
    if not os.path.exists(nasa_bioscience_processed1.csv):
        raise FileNotFoundError(f"CSV not found: {nasa_bioscience_processed1.csv}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows from {csv_path}")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_samples, dim)")
    if len(df) != embeddings.shape[0]:
        raise ValueError(f"Number of rows in CSV ({len(df)}) != number of embeddings ({embeddings.shape[0]})")

    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    # Ensure required columns exist
    required_cols = {"Title", "Abstract", "PCMID", "Link", "Summary"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Fill NaNs in summary with empty string
    df["Summary"] = df["Summary"].fillna("").astype(str)
    # Standardize title to string
    df["Title"] = df["Title"].fillna("").astype(str)
    return df.reset_index(drop=True), embeddings


def extract_result_sentences_from_summary(summary: str, max_results: int = 2) -> List[str]:
    """
    Heuristic extractor for 'Results/Outcomes' from an LLM-generated summary.

    Approach:
      - Split summary into sentences.
      - Prefer sentences containing result keywords (result, find, demonstrate, show, suggests, imply, indicate).
      - If none match, return first sentence(s) as fallback.

    Args:
        summary: text summary from LLM
        max_results: max result items to extract

    Returns:
        list of result strings (short)
    """
    import re

    if not summary or not isinstance(summary, str) or summary.strip() == "":
        return []

    # Very simple sentence splitter; adequate for short LLM summaries
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    keywords = ["result", "results", "found", "finding", "show", "demonstrate", "demonstrates",
                "suggest", "suggests", "indicate", "indicates", "observed", "conclude", "concludes"]
    selected = []
    for s in sentences:
        lower = s.lower()
        if any(k in lower for k in keywords):
            selected.append(s.strip())
        if len(selected) >= max_results:
            break

    # Fallback: take first sentences if none matched
    if not selected:
        selected = [s.strip() for s in sentences[:max_results] if s.strip()]

    # Further trim sentences to reasonable length
    trimmed = []
    for s in selected:
        if len(s) > 300:
            # truncate gently
            trimmed.append(s[:297].rstrip() + "...")
        else:
            trimmed.append(s)
    return trimmed


class KnowledgeGraphBuilder:
    """
    Build and query a knowledge graph from paper data + embeddings.

    Graph conventions:
      - Node attribute 'type' in {'paper', 'concept', 'result'}
      - Paper nodes: id = "paper:<PCMID>" (if PCMID exists) else "paper:<index>"
      - Concept nodes: id = "concept:<cluster_id>"
      - Result nodes: id = "result:<uuid>" or "result:<paper_pcmid>:<i>"

    Methods are designed to be called in sequence:
      1. __init__
      2. cluster_embeddings(...)
      3. build_graph(...)
      4. optionally link_similar_papers(...)
      5. export_graph_json(...)
    """

    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        Args:
            df: DataFrame with Title, Abstract, PCMID, Link, Summary
            embeddings: numpy array (n, d)
        """
        self.df = df.copy().reset_index(drop=True)
        self.embeddings = embeddings
        self.n, self.dim = embeddings.shape
        self.graph = nx.DiGraph()  # directed graph
        self.cluster_labels: Optional[np.ndarray] = None
        self.concept_centroids: Optional[np.ndarray] = None
        # Add paper nodes immediately
        self._add_paper_nodes()
        logger.info("KnowledgeGraphBuilder initialized.")

    def _paper_node_id(self, idx: int) -> str:
        pcmid = str(self.df.at[idx, "PCMID"]) if "PCMID" in self.df.columns else ""
        if pcmid and str(pcmid).strip():
            return f"paper:{pcmid}"
        else:
            return f"paper:idx{idx}"

    def _add_paper_nodes(self) -> None:
        """Insert paper nodes into self.graph with metadata."""
        for i, row in self.df.iterrows():
            node_id = self._paper_node_id(i)
            self.graph.add_node(node_id, type="paper",
                                title=row["Title"],
                                abstract=row.get("Abstract", ""),
                                pcmid=row["PCMID"] if "PCMID" in row else None,
                                link=row.get("Link", ""),
                                summary=row.get("Summary", ""),
                                idx=int(i))
        logger.info(f"Added {len(self.df)} paper nodes to the graph.")

    def cluster_embeddings(self, n_clusters: int = 20, random_state: int = 42,
                           use_kmeanspp: bool = True) -> np.ndarray:
        """
        Cluster the paper embeddings to form Concept nodes.

        Args:
            n_clusters: number of clusters (concepts). Choose based on dataset size.
            random_state: for reproducibility
            use_kmeanspp: whether to use k-means++ initialization

        Returns:
            cluster_labels: array of cluster ids (len == n_papers)
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        logger.info(f"Clustering embeddings into {n_clusters} clusters...")
        km = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++' if use_kmeanspp else 'random')
        self.cluster_labels = km.fit_predict(self.embeddings)
        # compute centroids for future use
        self.concept_centroids = km.cluster_centers_
        logger.info("Clustering completed.")
        return self.cluster_labels

    def create_concept_nodes(self, label_prefix: str = "Concept") -> None:
        """
        Create concept nodes for each cluster label (requires cluster_embeddings called first).

        Concept node attributes:
            - type='concept'
            - concept_id (cluster integer)
            - name: auto-generated like 'Concept 3'
            - size: number of papers in cluster (added)
        """
        if self.cluster_labels is None:
            raise RuntimeError("No cluster_labels found. Call cluster_embeddings() first.")
        cluster_ids, counts = np.unique(self.cluster_labels, return_counts=True)
        for cid, count in zip(cluster_ids, counts):
            node_id = f"concept:{cid}"
            self.graph.add_node(node_id, type="concept",
                                concept_id=int(cid),
                                name=f"{label_prefix} {cid}",
                                size=int(count))
        logger.info(f"Created {len(cluster_ids)} concept nodes.")

    def link_papers_to_concepts(self) -> None:
        """
        Link each paper -> its concept node (Paper -> Concept) with similarity score
        equal to cosine_similarity(paper_embedding, concept_centroid).
        """
        if self.cluster_labels is None or self.concept_centroids is None:
            raise RuntimeError("Clusters not computed. Call cluster_embeddings() first.")
        # compute similarities
        sims = cosine_similarity(self.embeddings, self.concept_centroids)  # (n, k)
        for i in range(self.n):
            cid = int(self.cluster_labels[i])
            paper_node = self._paper_node_id(i)
            concept_node = f"concept:{cid}"
            sim = float(sims[i, cid])
            self.graph.add_edge(paper_node, concept_node, type="has_concept", weight=sim)
        logger.info("Linked papers to their concept nodes.")

    def generate_result_nodes_from_summaries(self, max_results_per_paper: int = 2) -> List[str]:
        """
        Parse summaries to create Result nodes. Returns list of result node IDs created.

        Heuristic: extract 1-2 'result' sentences per summary using extract_result_sentences_from_summary.
        Each result node references its origin paper (paper idx/PCMID).
        """
        created = []
        for i, row in self.df.iterrows():
            summary = str(row.get("Summary", "") or "")
            result_texts = extract_result_sentences_from_summary(summary, max_results=max_results_per_paper)
            for j, text in enumerate(result_texts):
                # create deterministic id using pcmid if present
                pcmid = str(row["PCMID"]) if "PCMID" in row else ""
                rid = f"result:{pcmid}:{j}" if pcmid and pcmid.strip() else f"result:idx{i}:{j}"
                # avoid duplicates
                if self.graph.has_node(rid):
                    continue
                # node attributes
                self.graph.add_node(rid, type="result", text=text, origin_paper=self._paper_node_id(i))
                # Link paper->result
                self.graph.add_edge(self._paper_node_id(i), rid, type="mentions_result")
                created.append(rid)
        logger.info(f"Created {len(created)} result nodes from summaries.")
        return created

    def link_concepts_to_results(self, top_k_papers: int = 5) -> None:
        """
        For each concept, choose top_k_papers nearest to the concept centroid and attach
        their result nodes (if any) to the concept node (Concept -> Result) with weight derived
        from similarity of the paper to centroid.

        This links conceptual themes to outcome/result nodes extracted from summaries.
        """
        if self.concept_centroids is None:
            raise RuntimeError("Concept centroids not computed. Call cluster_embeddings() first.")
        # compute similarity of every paper to every centroid
        sims = cosine_similarity(self.embeddings, self.concept_centroids)  # (n, k)
        k = self.concept_centroids.shape[0]
        for cid in range(k):
            concept_node = f"concept:{cid}"
            # find top papers indices by similarity to centroid
            top_indices = np.argsort(-sims[:, cid])[:top_k_papers]
            for idx in top_indices:
                paper_node = self._paper_node_id(int(idx))
                sim = float(sims[idx, cid])
                # find result nodes that originate from this paper
                # result nodes are those with attribute type=='result' and origin_paper == paper_node
                for node, data in list(self.graph.nodes(data=True)):
                    if data.get("type") == "result" and data.get("origin_paper") == paper_node:
                        result_node = node
                        # Add edge concept -> result
                        self.graph.add_edge(concept_node, result_node, type="concept_has_result", weight=sim)
        logger.info("Linked concepts to result nodes based on top-k papers per concept.")

    def link_similar_papers(self, top_k: int = 5, min_similarity: float = 0.65) -> None:
        """
        Create symmetric Paper -> Paper similarity edges for top_k neighbors per paper,
        filtered by a minimum cosine similarity threshold.

        Edge attributes:
            - type="similar_paper"
            - weight = similarity score

        Args:
            top_k: how many neighbors to attach (per-paper)
            min_similarity: minimum cosine similarity to create an edge
        """
        logger.info("Computing paper-paper similarities...")
        # Efficient approach: compute cosine similarity matrix in blocks if necessary.
        # For moderate sizes, compute full similarity matrix.
        sims = cosine_similarity(self.embeddings)  # (n, n)
        for i in range(self.n):
            sims_i = sims[i]
            # exclude self
            sims_i[i] = -1.0
            # get top_k indices
            top_idx = np.argsort(-sims_i)[:top_k]
            for j in top_idx:
                sim = float(sims_i[j])
                if sim >= min_similarity:
                    a = self._paper_node_id(i)
                    b = self._paper_node_id(int(j))
                    # create directed edge a->b; clients can interpret as undirected if desired
                    self.graph.add_edge(a, b, type="similar_paper", weight=sim)
        logger.info("Linked similar papers based on embedding similarity.")

    def summarize_concept(self, concept_id: int, top_n_titles: int = 5) -> Dict[str, Any]:
        """
        Produce a brief summary for a concept using top nearest paper titles and aggregated TF-IDF keywords.

        Returns:
            {
                'concept_id': concept_id,
                'top_titles': [...],
                'top_keywords': [...]
            }
        """
        if self.concept_centroids is None or self.cluster_labels is None:
            raise RuntimeError("Clustering not performed.")

        # papers in this cluster
        indices = np.where(self.cluster_labels == concept_id)[0]
        if len(indices) == 0:
            return {"concept_id": concept_id, "top_titles": [], "top_keywords": []}

        # Top titles (closest to centroid)
        centroid = self.concept_centroids[concept_id].reshape(1, -1)
        dists = pairwise_distances(self.embeddings[indices], centroid, metric="cosine").reshape(-1)
        order = np.argsort(dists)
        top_indices = indices[order][:top_n_titles]
        top_titles = [self.df.at[int(idx), "Title"] for idx in top_indices]

        # Keywords: TF-IDF over abstracts+summaries of cluster
        docs = (self.df.loc[indices, "Abstract"].fillna("") + " " + self.df.loc[indices, "Summary"].fillna("")).tolist()
        if len(docs) >= 1:
            vect = TfidfVectorizer(max_features=30, stop_words='english', ngram_range=(1, 2))
            X = vect.fit_transform(docs)
            # average tfidf scores across docs and pick top terms
            mean_scores = X.mean(axis=0).A1
            terms = np.array(vect.get_feature_names_out())
            top_term_idx = np.argsort(-mean_scores)[:10]
            top_keywords = terms[top_term_idx].tolist()
        else:
            top_keywords = []

        return {"concept_id": concept_id, "top_titles": top_titles, "top_keywords": top_keywords}

    def query_by_paper_title(self, title_substring: str, fuzzy: bool = False) -> Dict[str, Any]:
        """
        Query graph by matching paper title substring (case-insensitive).
        Returns node info and immediate neighbors (edges and nodes).

        Args:
            title_substring: substring to search in titles
            fuzzy: if True, use simple lowercase inclusion; (reserved for future fuzzy search)

        Returns:
            dict with matching papers and their neighbors
        """
        title_substring = title_substring.lower()
        matches = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "paper":
                title = str(data.get("title", "")).lower()
                if title_substring in title:
                    # gather neighbors
                    neighbors_out = list(self.graph[node].items())
                    neighbors_in = list(self.graph.pred[node].items())
                    matches.append({"node": node, "meta": data, "out_edges": neighbors_out, "in_edges": neighbors_in})
        return {"query": title_substring, "results": matches}

    def query_by_concept(self, concept_id: int) -> Dict[str, Any]:
        """
        Return concept node attributes, member papers, top keywords, and connected results.
        """
        concept_node = f"concept:{concept_id}"
        if not self.graph.has_node(concept_node):
            return {"error": f"Concept {concept_id} not found"}
        data = self.graph.nodes[concept_node]
        # member papers are those with an edge paper->concept
        members = [u for u, v, ed in self.graph.in_edges(concept_node, data=True) if self.graph.nodes[u].get("type") == "paper"]
        # results linked to concept
        results = [v for u, v, ed in self.graph.out_edges(concept_node, data=True) if self.graph.nodes[v].get("type") == "result"]
        summary = self.summarize_concept(concept_id)
        return {"concept": concept_node, "meta": data, "member_papers": members, "linked_results": results, "summary": summary}

    def export_graph_json(self, node_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export graph into a JSON-serializable dict with 'nodes' and 'links' arrays for frontend visualizations.

        Args:
            node_filter: optional list of node IDs to include only a subgraph (if None, include all)

        Returns:
            dict with 'nodes' and 'links'
        """
        if node_filter is not None:
            nodes = [n for n in self.graph.nodes() if n in set(node_filter)]
        else:
            nodes = list(self.graph.nodes())

        nodes_out = []
        id_map = {}  # map node id to integer index for frontend
        for i, n in enumerate(nodes):
            id_map[n] = i
            data = dict(self.graph.nodes[n])
            data["id"] = n
            nodes_out.append(data)

        links_out = []
        for u, v, ed in self.graph.edges(data=True):
            if u in id_map and v in id_map:
                link = {
                    "source": id_map[u],
                    "target": id_map[v],
                    "type": ed.get("type"),
                    "weight": float(ed.get("weight", 1.0))
                }
                links_out.append(link)

        return {"nodes": nodes_out, "links": links_out}

    def save_graph_to_file(self, path: str) -> None:
        """
        Save networkx graph to JSON file (export_graph_json).
        """
        obj = self.export_graph_json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph exported to {path}")

    def get_networkx_graph(self) -> nx.DiGraph:
        """Return underlying networkx graph for advanced operations."""
        return self.graph


# Example of usage (to be invoked in application code or tests)
def build_kg_pipeline(csv_path: str, embeddings_path: str,
                      n_concepts: int = 20,
                      top_k_papers_per_concept: int = 5,
                      similar_paper_topk: int = 5,
                      similar_paper_minsim: float = 0.65) -> KnowledgeGraphBuilder:
    """
    Convenience pipeline to load data, cluster embeddings, create nodes and link them.

    Returns:
        KnowledgeGraphBuilder instance with graph built.
    """
    df, embeddings = load_csv_embeddings(csv_path, embeddings_path)
    builder = KnowledgeGraphBuilder(df, embeddings)
    builder.cluster_embeddings(n_clusters=n_concepts)
    builder.create_concept_nodes()
    builder.link_papers_to_concepts()
    builder.generate_result_nodes_from_summaries(max_results_per_paper=2)
    builder.link_concepts_to_results(top_k_papers=top_k_papers_per_concept)
    builder.link_similar_papers(top_k=similar_paper_topk, min_similarity=similar_paper_minsim)
    return builder


# If this module is run directly, a small example dry-run (assuming example files exist)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build knowledge graph from CSV + embeddings")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file (Title, Abstract, PCMID, Link, Summary)")
    parser.add_argument("--emb", type=str, required=True, help=".npy file of embeddings")
    parser.add_argument("--out", type=str, default="kg_export.json", help="Output JSON path")
    parser.add_argument("--n_concepts", type=int, default=20)
    args = parser.parse_args()

    kg = build_kg_pipeline(args.csv, args.emb, n_concepts=args.n_concepts)
    kg.save_graph_to_file(args.out)
    logger.info("Done.")
