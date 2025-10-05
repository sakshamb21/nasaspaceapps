# extract_pubmed_data.py
import pandas as pd
import Bio.Entrez as Entrez
from tqdm import tqdm
import time

# REQUIRED: Enter your email (NCBI needs this for API usage)
Entrez.email = "sakshamavailable@gmail.com"

def extract_pmid_from_url(url: str) -> str:
    """
    Extract PMID from a typical PubMed link like:
    https://www.ncbi.nlm.nih.gov/pubmed/12345678
    """
    try:
        return url.rstrip("/").split("/")[-1]
    except Exception:
        return None

def fetch_pubmed_metadata(pmid: str) -> dict:
    """
    Fetch metadata and abstract from PubMed using Entrez API.
    Returns dictionary with fields.
    """
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
        records = Entrez.read(handle)
        article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

        title = str(article.get("ArticleTitle", ""))
        abstract = " ".join([p for p in article.get("Abstract", {}).get("AbstractText", [])])
        journal = str(article.get("Journal", {}).get("Title", ""))
        year = str(article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", ""))
        authors = []
        for a in article.get("AuthorList", []):
            if "LastName" in a and "ForeName" in a:
                authors.append(f"{a['ForeName']} {a['LastName']}")
        authors = ", ".join(authors)

        return {
            "PMID": pmid,
            "Title": title,
            "Authors": authors,
            "Year": year,
            "Journal": journal,
            "Abstract": abstract
        }
    except Exception as e:
        return {
            "PMID": pmid,
            "Title": "",
            "Authors": "",
            "Year": "",
            "Journal": "",
            "Abstract": "",
            "Error": str(e)
        }

def build_dataset(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pmid = extract_pmid_from_url(row["Link"])
        if not pmid:
            continue
        meta = fetch_pubmed_metadata(pmid)
        meta["Link"] = row["Link"]
        results.append(meta)
        time.sleep(0.34)  # To avoid hitting NCBI rate limits (~3 req/sec)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Processed dataset saved to {output_csv}")

if __name__ == "__main__":
    build_dataset("SB_publication_PMC.csv", "nasa_bioscience_processed.csv")
