# graph_integration.py
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from neo4j_client import run_cypher

# Try to load scispaCy; fallback to spaCy if missing
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except ImportError:
    nlp = None
    print("⚠️ spaCy not installed, entity extraction disabled")


def extract_topics_and_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract biomedical entities (topics, chemicals, genes, etc.)
    using scispaCy if available. Falls back to simple noun chunks.
    """
    if not nlp or not text:
        return {"topics": [], "entities": []}

    doc = nlp(text)

    topics = []
    entities = []

    # Named Entities
    for ent in doc.ents:
        entities.append(ent.text)

    # Key Noun Chunks as "topics"
    for chunk in doc.noun_chunks:
        topics.append(chunk.text)

    return {
        "topics": list(set(topics)),
        "entities": list(set(entities)),
    }


def merge_study(study_id: str, title: str, abstract: str):
    query = """
    MERGE (s:Study {id: $id})
    SET s.title = coalesce($title, s.title),
        s.abstract = coalesce($abstract, s.abstract),
        s.lastUpdated = datetime()
    RETURN s
    """
    return run_cypher(query, {"id": study_id, "title": title, "abstract": abstract}, write=True)


def merge_topic(study_id: str, topic: str):
    query = """
    MERGE (t:Topic {name: $topic})
    WITH t
    MATCH (s:Study {id: $study_id})
    MERGE (s)-[:HAS_TOPIC]->(t)
    RETURN t
    """
    return run_cypher(query, {"study_id": study_id, "topic": topic}, write=True)


def merge_entity(study_id: str, entity: str):
    query = """
    MERGE (e:Entity {name: $entity})
    WITH e
    MATCH (s:Study {id: $study_id})
    MERGE (s)-[:MENTIONS]->(e)
    RETURN e
    """
    return run_cypher(query, {"study_id": study_id, "entity": entity}, write=True)


def create_query_node(query_text: str, purpose: str = "general"):
    """
    Represent a user query in the graph for traceability.
    """
    qid = str(uuid.uuid4())
    query = """
    MERGE (q:Query {id: $id})
    SET q.text = $text,
        q.purpose = $purpose,
        q.timestamp = datetime()
    RETURN q
    """
    return run_cypher(
        query,
        {"id": qid, "text": query_text, "purpose": purpose},
        write=True,
    )


def ingest_summarizer_output(study: Dict[str, Any]):
    """
    Ingest publication data into the graph.
    Expects dict: {id, title, abstract}
    """
    study_id = study["id"]
    title = study.get("title")
    abstract = study.get("abstract", "")

    # Merge study node
    merge_study(study_id, title, abstract)

    # Extract concepts
    extracted = extract_topics_and_entities(abstract)
    for topic in extracted["topics"]:
        merge_topic(study_id, topic)
    for entity in extracted["entities"]:
        merge_entity(study_id, entity)

    return {"status": "ok", "study": study_id}
