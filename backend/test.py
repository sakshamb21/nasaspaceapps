
import requests

BASE_URL = "http://127.0.0.1:8000/api"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.json())

def test_search():
    r = requests.get(f"{BASE_URL}/search", params={"query": "bone density in space", "top_k": 3})
    print("Search results:")
    for row in r.json():
        print("-", row)

def test_summarize():
    payload = {
        "query": "bone density in space",
        "audience": "scientists",
        "papers": ["12345", "67890"],  # dummy pcms or IDs
    }
    r = requests.post(f"{BASE_URL}/summarize", json=payload)
    print("Summarize:", r.json())
    return r.json()

def test_graph_update(summary_result):
    payload = {
        "query": summary_result.get("query", "bone density in space"),
        "audience": summary_result.get("audience", "scientists"),
        "summary": summary_result.get("summary", "dummy summary"),
        "sources": summary_result.get("papers", ["12345"])
    }
    r = requests.post(f"{BASE_URL}/graph/update", json=payload)
    print("Graph update:", r.json())

def test_graph_fetch()
    r = requests.get(f"{BASE_URL}/graph", params={"pcmid": "12345", "radius": 2})
    print("Graph fetch:", r.json())

if __name__ == "__main__":
    #test_health()
    #test_search()
    summary_result = test_summarize()
    test_graph_update(summary_result)
    test_graph_fetch()
