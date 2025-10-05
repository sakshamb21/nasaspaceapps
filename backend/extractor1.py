# extract_pubmed_pmc_data_final.py
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import time
import xml.etree.ElementTree as ET

# REQUIRED: your email (NCBI API requirement)
Entrez.email = "sakshamavailable@gmail.com"

def extract_ids(url: str) -> dict:
    """
    Extract PMID or PMCID from URL.
    """
    url = url.rstrip("/")
    if "pubmed" in url:
        return {"PMID": url.split("/")[-1], "PMCID": ""}
    elif "pmc" in url:
        return {"PMID": "", "PMCID": url.split("/")[-1]}
    return {"PMID": "", "PMCID": ""}

def pmc_to_pmid(pmcid: str) -> str:
    """
    Resolve PMC ID → PMID using Entrez.
    Returns PMID string or "" if not found.
    """
    try:
        handle = Entrez.elink(dbfrom="pmc", db="pubmed", id=pmcid)
        records = Entrez.read(handle)
        links = records[0]["LinkSetDb"]
        if links and "Link" in links[0]:
            return links[0]["Link"][0]["Id"]
    except Exception:
        return ""
    return ""

def fetch_pubmed_metadata(pmid: str) -> dict:
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        records = Entrez.read(handle, validate=False)
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
            "PMCID": "",
            "Title": title,
            "Authors": authors,
            "Year": year,
            "Journal": journal,
            "Abstract": abstract,
            "SourceDB": "pubmed",
            "Error": ""
        }
    except Exception as e:
        return {
            "PMID": pmid, "PMCID": "", "Title": "", "Authors": "",
            "Year": "", "Journal": "", "Abstract": "",
            "SourceDB": "pubmed", "Error": str(e)
        }

def fetch_pmc_metadata_raw(pmcid: str) -> dict:
    """
    Fallback: direct PMC XML parse if PMID cannot be resolved.
    """
    try:
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
        xml_data = handle.read()
        root = ET.fromstring(xml_data)

        # Title
        title = ""
        title_el = root.find(".//article-title")
        if title_el is not None:
            title = "".join(title_el.itertext())

        # Abstract
        abstract = ""
        abs_el = root.find(".//abstract")
        if abs_el is not None:
            abstract = " ".join("".join(p.itertext()) for p in abs_el.findall(".//p"))

        # Authors
        authors = []
        for contrib in root.findall(".//contrib[@contrib-type='author']"):
            fname = contrib.findtext(".//given-names", default="")
            lname = contrib.findtext(".//surname", default="")
            if fname or lname:
                authors.append(f"{fname} {lname}".strip())
        authors = ", ".join(authors)

        # Journal
        journal = ""
        journal_el = root.find(".//journal-title")
        if journal_el is not None:
            journal = "".join(journal_el.itertext())

        # Year
        year = ""
        pubdate_el = root.find(".//pub-date/year")
        if pubdate_el is not None:
            year = pubdate_el.text

        return {
            "PMID": "",
            "PMCID": pmcid,
            "Title": title,
            "Authors": authors,
            "Year": year,
            "Journal": journal,
            "Abstract": abstract,
            "SourceDB": "pmc",
            "Error": ""
        }
    except Exception as e:
        return {
            "PMID": "", "PMCID": pmcid, "Title": "", "Authors": "",
            "Year": "", "Journal": "", "Abstract": "",
            "SourceDB": "pmc", "Error": str(e)
        }

def build_dataset(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ids = extract_ids(row["Link"])

        if ids["PMID"]:
            meta = fetch_pubmed_metadata(ids["PMID"])
        elif ids["PMCID"]:
            pmid = pmc_to_pmid(ids["PMCID"])
            if pmid:  # Found mapping → fetch via PubMed
                meta = fetch_pubmed_metadata(pmid)
                meta["PMCID"] = ids["PMCID"]
            else:  # No mapping → fallback raw PMC parsing
                meta = fetch_pmc_metadata_raw(ids["PMCID"])
        else:
            meta = {
                "PMID": "", "PMCID": "", "Title": "", "Authors": "",
                "Year": "", "Journal": "", "Abstract": "",
                "SourceDB": "", "Error": "Could not detect PMID/PMCID"
            }

        meta["Link"] = row["Link"]
        results.append(meta)
        time.sleep(0.34)  # respect NCBI rate limit

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Processed dataset saved to {output_csv}")

if __name__ == "__main__":
    build_dataset("SB_publication_PMC.csv", "nasa_bioscience_processed1.csv")