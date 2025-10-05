import networkx as nx

# global in-memory graph
GRAPH = nx.DiGraph()

def update_graph(query, audience, summary, sources):
    """
    Add/update nodes and edges for the graph.
    """
    # Add the query node if not exists
    if not GRAPH.has_node(query):
        GRAPH.add_node(query, type="topic")

    # Add summary node
    summary_node = f"{query}_{audience}"
    GRAPH.add_node(summary_node, type="summary", text=summary)
    GRAPH.add_edge(query, summary_node, relation="summarized_for")

    # Add source papers
    for src in sources:
        if not GRAPH.has_node(src):
            GRAPH.add_node(src, type="paper")
        GRAPH.add_edge(summary_node, src, relation="based_on")

    return {"message": "Graph updated", "nodes": len(GRAPH.nodes), "edges": len(GRAPH.edges)}


def fetch_graph(center, radius=2):
    """
    Return a subgraph around a given node.
    """
    if not GRAPH.has_node(center):
        return {"error": "Node not found"}

    # Collect neighbors within radius
    nodes = set([center])
    frontier = [center]
    for _ in range(radius):
        next_frontier = []
        for node in frontier:
            neighbors = list(GRAPH.successors(node)) + list(GRAPH.predecessors(node))
            for n in neighbors:
                if n not in nodes:
                    nodes.add(n)
                    next_frontier.append(n)
        frontier = next_frontier

    subgraph = GRAPH.subgraph(nodes)

    # Convert to JSON format (nodes + links)
    nodes_data = [{"id": n, "type": subgraph.nodes[n].get("type", "unknown")} for n in subgraph.nodes]
    edges_data = [{"source": u, "target": v, "relation": d.get("relation", "")} for u, v, d in subgraph.edges(data=True)]

    return {"nodes": nodes_data, "edges": edges_data}