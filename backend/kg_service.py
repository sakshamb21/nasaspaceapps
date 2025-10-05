"""
Neo4j Knowledge Graph Service
Handles all Neo4j database operations for the knowledge graph.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kg_service")

# ============================================================================
# Neo4j Driver Management
# ============================================================================

@lru_cache(maxsize=1)
def get_neo4j_driver():
    """Initialize and return a Neo4j driver (cached)."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info(f"Neo4j driver initialized: {NEO4J_URI}")
        return driver
    except AuthError:
        logger.error("Neo4j authentication failed")
        raise ValueError("Neo4j authentication failed. Check credentials.")
    except ServiceUnavailable:
        logger.error("Neo4j service unavailable")
        raise ValueError("Neo4j service unavailable. Ensure Neo4j is running.")
    except Exception as e:
        logger.exception("Failed to initialize Neo4j driver")
        raise RuntimeError(f"Neo4j initialization failed: {e}")

# ============================================================================
# Knowledge Graph Creation
# ============================================================================

def create_knowledge_graph(kg_data: Dict[str, Any], query_text: str) -> bool:
    """
    Create nodes and relationships in Neo4j from extracted knowledge graph data.
    
    Args:
        kg_data: Dictionary with 'entities' and 'relationships' lists
        query_text: The original user query
        
    Returns:
        True if successful, False otherwise
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # 1. Create or merge the query node
            session.run(
                """
                MERGE (q:Query {text: $text})
                ON CREATE SET q.timestamp = timestamp(), q.count = 1
                ON MATCH SET q.count = q.count + 1, q.last_accessed = timestamp()
                RETURN q
                """,
                text=query_text
            )
            
            # 2. Create entity nodes
            for entity in kg_data.get("entities", []):
                label = entity.get("label", "Entity")
                entity_id = entity.get("id", "")
                properties = entity.get("properties", {})
                
                if not entity_id:
                    logger.warning(f"Skipping entity with no ID: {entity}")
                    continue
                
                # Sanitize label (remove spaces, special chars)
                label = "".join(c for c in label if c.isalnum())
                
                try:
                    # Use parameterized query for safety
                    query = f"""
                    MERGE (e:{label} {{id: $id}})
                    ON CREATE SET e += $properties, e.created = timestamp()
                    ON MATCH SET e += $properties, e.updated = timestamp()
                    RETURN e
                    """
                    
                    session.run(query, id=entity_id, properties=properties)
                except Exception as e:
                    logger.error(f"Failed to create entity {entity_id}: {e}")
                    continue
            
            # 3. Create relationships
            for rel in kg_data.get("relationships", []):
                from_id = rel.get("from", "")
                to_id = rel.get("to", "")
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                
                if not from_id or not to_id:
                    logger.warning(f"Skipping relationship with missing IDs: {rel}")
                    continue
                
                # Sanitize relationship type
                rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type).upper()
                
                try:
                    cypher = f"""
                    MATCH (a {{id: $from_id}})
                    MATCH (b {{id: $to_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    ON CREATE SET r += $properties, r.created = timestamp()
                    ON MATCH SET r += $properties, r.updated = timestamp()
                    RETURN r
                    """
                    
                    session.run(cypher, from_id=from_id, to_id=to_id, properties=properties)
                except Exception as e:
                    logger.error(f"Failed to create relationship {from_id}->{to_id}: {e}")
                    continue
            
            # 4. Link query to entities
            for entity in kg_data.get("entities", []):
                entity_id = entity.get("id")
                if entity_id:
                    try:
                        session.run(
                            """
                            MATCH (q:Query {text: $query_text})
                            MATCH (e {id: $entity_id})
                            MERGE (q)-[:EXTRACTED]->(e)
                            """,
                            query_text=query_text,
                            entity_id=entity_id
                        )
                    except Exception as e:
                        logger.error(f"Failed to link query to entity {entity_id}: {e}")
        
        logger.info(f"Knowledge graph created: {len(kg_data.get('entities', []))} entities, "
                   f"{len(kg_data.get('relationships', []))} relationships")
        return True
        
    except Exception as e:
        logger.exception("Failed to create knowledge graph")
        return False

# ============================================================================
# Query Functions
# ============================================================================

def get_entity_by_id(entity_id: str) -> Optional[Dict[str, Any]]:
    """Get an entity and its relationships by ID."""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e {id: $entity_id})
                OPTIONAL MATCH (e)-[r]->(connected)
                RETURN e, labels(e) as labels,
                       collect(DISTINCT {
                           type: type(r), 
                           target: connected,
                           target_labels: labels(connected),
                           properties: properties(r)
                       }) as outgoing
                """,
                entity_id=entity_id
            )
            
            record = result.single()
            if not record:
                return None
            
            entity = dict(record["e"])
            entity["labels"] = record["labels"]
            entity["relationships"] = {
                "outgoing": [r for r in record["outgoing"] if r["target"] is not None]
            }
            
            # Get incoming relationships
            result = session.run(
                """
                MATCH (source)-[r]->(e {id: $entity_id})
                RETURN collect(DISTINCT {
                    type: type(r),
                    source: source,
                    source_labels: labels(source),
                    properties: properties(r)
                }) as incoming
                """,
                entity_id=entity_id
            )
            
            record = result.single()
            if record:
                entity["relationships"]["incoming"] = [
                    r for r in record["incoming"] if r["source"] is not None
                ]
            
            return entity
            
    except Exception as e:
        logger.exception(f"Failed to get entity {entity_id}")
        return None

def search_entities(
    label: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Search for entities by label and/or name."""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Build dynamic query
            if label and name:
                query = f"""
                MATCH (e:{label})
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e, labels(e) as labels
                LIMIT $limit
                """
                result = session.run(query, name=name, limit=limit)
            elif label:
                query = f"""
                MATCH (e:{label})
                RETURN e, labels(e) as labels
                LIMIT $limit
                """
                result = session.run(query, limit=limit)
            elif name:
                query = """
                MATCH (e)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e, labels(e) as labels
                LIMIT $limit
                """
                result = session.run(query, name=name, limit=limit)
            else:
                query = """
                MATCH (e)
                RETURN e, labels(e) as labels
                LIMIT $limit
                """
                result = session.run(query, limit=limit)
            
            entities = []
            for record in result:
                entity = dict(record["e"])
                entity["labels"] = record["labels"]
                entities.append(entity)
            
            return entities
            
    except Exception as e:
        logger.exception("Failed to search entities")
        return []

def get_query_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent queries and their extracted entities."""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run(
                """
                MATCH (q:Query)
                OPTIONAL MATCH (q)-[:EXTRACTED]->(e)
                RETURN q, collect(DISTINCT {
                    id: e.id,
                    labels: labels(e),
                    name: e.name
                }) as entities
                ORDER BY q.timestamp DESC
                LIMIT $limit
                """,
                limit=limit
            )
            
            queries = []
            for record in result:
                query = dict(record["q"])
                query["entities"] = [e for e in record["entities"] if e["id"] is not None]
                queries.append(query)
            
            return queries
            
    except Exception as e:
        logger.exception("Failed to get query history")
        return []

def get_graph_statistics() -> Dict[str, Any]:
    """Get statistics about the knowledge graph."""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Get node counts by label
            result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
                """
            )
            node_counts = {record["label"]: record["count"] for record in result}
            
            # Get relationship counts by type
            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
                """
            )
            rel_counts = {record["type"]: record["count"] for record in result}
            
            # Get total counts
            result = session.run(
                """
                MATCH (n)
                WITH count(n) as node_count
                MATCH ()-[r]->()
                RETURN node_count, count(r) as rel_count
                """
            )
            record = result.single()
            
            return {
                "total_nodes": record["node_count"] if record else 0,
                "total_relationships": record["rel_count"] if record else 0,
                "nodes_by_label": node_counts,
                "relationships_by_type": rel_counts
            }
            
    except Exception as e:
        logger.exception("Failed to get graph statistics")
        return {
            "total_nodes": 0,
            "total_relationships": 0,
            "nodes_by_label": {},
            "relationships_by_type": {}
        }

def clear_knowledge_graph() -> Dict[str, Any]:
    """Clear all nodes and relationships from the knowledge graph."""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Get counts before deletion
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            
            return {
                "nodes_deleted": node_count,
                "relationships_deleted": rel_count
            }
            
    except Exception as e:
        logger.exception("Failed to clear knowledge graph")
        raise