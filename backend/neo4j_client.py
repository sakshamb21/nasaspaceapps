# neo4j_client.py
from neo4j import GraphDatabase, basic_auth, READ_ACCESS, WRITE_ACCESS
from typing import List, Dict, Any, Optional, Union

# Singleton driver
_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri = "bolt://localhost:7687"  # Change if running remotely
        user = "neo4j"
        password = "password"
        _driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_cypher(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    write: bool = False,
    as_dict: bool = True,
) -> List[Union[Dict[str, Any], Any]]:
    """
    Run a Cypher query safely.

    Args:
        query: Cypher string
        params: parameter dictionary
        write: whether to use WRITE access
        as_dict: if True, return dict(row); if False, return raw Neo4j Record

    Returns:
        List of dicts (if as_dict=True) or raw records
    """
    driver = get_driver()
    session_mode = WRITE_ACCESS if write else READ_ACCESS
    results = []

    with driver.session(default_access_mode=session_mode) as session:
        records = session.run(query, params or {})
        if as_dict:
            results = [dict(r.items()) for r in records]
        else:
            results = list(records)

    return results