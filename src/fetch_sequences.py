#!/usr/bin/env python
"""
Fetch a small, labelled subset of real protein sequences for the real-data
recovery-threshold test.

Output is SOURCE-AGNOSTIC so the rest of the pipeline never cares where the
sequences came from:

    <outdir>/seqs.fasta        one record per sequence, id == accession
    <outdir>/metadata.tsv      columns: id <TAB> family <TAB> group

  * family  -> becomes the integer label y (shared label space across groups)
  * group   -> the covariate-shift axis. SOURCE vs TARGET split is done later
               in precompute_real_embeddings.py. For UniProt this is a taxon;
               for MGnify it is a biome (MGnify metagenomic proteins do not
               carry clean per-protein taxonomy -- see HOW_TO_SEE_THE_DIP.md).

Two backends:

    --source uniprot   (recommended/default) clean Pfam family + taxonomy via
                       the UniProtKB REST API. Gives a genuine TAXONOMIC shift.
    --source mgnify    pulls from the MGnify proteins BigQuery public dataset.
                       Gives a BIOME shift. Requires `google-cloud-bigquery`
                       and GCP credentials; the table/columns assumed are
                       documented in fetch_mgnify() and easy to tweak.
"""
import argparse
import sys
import time

import requests

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def fetch_uniprot(families, groups, per_bucket, reviewed_only):
    """
    families: list of Pfam accessions, e.g. ["PF00069", "PF00005"]
    groups:   dict {group_name: taxonomy_id}, e.g. {"bacteria": 2, "archaea": 2157}
    Returns list of (accession, sequence, family, group).
    """
    records = []
    seen = set()
    for fam in families:
        for gname, taxid in groups.items():
            q = f"(xref:pfam-{fam}) AND (taxonomy_id:{taxid})"
            if reviewed_only:
                q += " AND (reviewed:true)"
            params = {
                "query": q,
                "fields": "accession,sequence",
                "format": "json",
                "size": 500,
            }
            url = UNIPROT_SEARCH
            got = 0
            print(f"[uniprot] {fam} / {gname} (taxid {taxid}) ...", flush=True)
            while url and got < per_bucket:
                r = requests.get(url, params=params, timeout=120)
                params = None  # cursor URL already encodes params after first call
                if r.status_code != 200:
                    print(f"  WARN http {r.status_code}: {r.text[:200]}", flush=True)
                    break
                for item in r.json().get("results", []):
                    acc = item["primaryAccession"]
                    seq = item.get("sequence", {}).get("value")
                    if not seq or acc in seen:
                        continue
                    seen.add(acc)
                    records.append((acc, seq, fam, gname))
                    got += 1
                    if got >= per_bucket:
                        break
                # UniProt paginates via a Link: rel="next" header
                url = r.links.get("next", {}).get("url")
                time.sleep(0.2)
            print(f"  -> {got} sequences", flush=True)
    return records


def fetch_uniprot_curation(families, per_bucket, taxid=None):
    """
    SwissProt -> distance-from-SwissProt shift, fetchable TODAY via the UniProt REST
    API (no GCP). Same Pfam families in both groups; the covariate-shift axis is
    curation status:

        group 'swissprot' = reviewed:true   (curated; the SOURCE)
        group 'trembl'    = reviewed:false  (computationally annotated; the "distance")

    This operationalizes the note "source looks like swissprot, target is a distance
    from swissprot" without metagenomics. For the more extreme distance (MGnify
    metagenomic proteins), use --source mgnify (needs BigQuery creds).

    taxid: optional taxonomy_id to hold the taxon fixed so the ONLY shift is
           curation status (recommended, e.g. 2 for Bacteria).
    Returns list of (accession, sequence, family, group).
    """
    records = []
    seen = set()
    curation_groups = {"swissprot": "true", "trembl": "false"}
    for fam in families:
        for gname, reviewed in curation_groups.items():
            q = f"(xref:pfam-{fam}) AND (reviewed:{reviewed})"
            if taxid is not None:
                q += f" AND (taxonomy_id:{taxid})"
            params = {
                "query": q,
                "fields": "accession,sequence",
                "format": "json",
                "size": 500,
            }
            url = UNIPROT_SEARCH
            got = 0
            print(f"[curation] {fam} / {gname} (reviewed:{reviewed}) ...", flush=True)
            while url and got < per_bucket:
                r = requests.get(url, params=params, timeout=120)
                params = None
                if r.status_code != 200:
                    print(f"  WARN http {r.status_code}: {r.text[:200]}", flush=True)
                    break
                for item in r.json().get("results", []):
                    acc = item["primaryAccession"]
                    seq = item.get("sequence", {}).get("value")
                    if not seq or acc in seen:
                        continue
                    seen.add(acc)
                    records.append((acc, seq, fam, gname))
                    got += 1
                    if got >= per_bucket:
                        break
                url = r.links.get("next", {}).get("url")
                time.sleep(0.2)
            print(f"  -> {got} sequences", flush=True)
    return records


def fetch_mgnify(families, groups, per_bucket):
    """
    MGnify proteins via the Google Cloud BigQuery public dataset
    (release 2024_04). Requires `pip install google-cloud-bigquery` and
    `gcloud auth application-default login` (or a service-account key).

    NOTE: column/table names below reflect the published MGnify-proteins
    BigQuery schema at time of writing. If a query errors, print the schema
    with `bq show <dataset>.<table>` and adjust the SELECT/WHERE here.

    groups: dict {group_name: biome_substring}, e.g.
            {"gut": "Human:Digestive system", "marine": "Environmental:Aquatic:Marine"}
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit(
            "MGnify backend needs google-cloud-bigquery:\n"
            "  pip install google-cloud-bigquery\n"
            "  gcloud auth application-default login"
        )

    client = bigquery.Client()
    # Public dataset; adjust if EBI bumps the release.
    PROTEINS = "mgnify-proteins.protein_database_2024_04.proteins"
    records = []
    for fam in families:
        for gname, biome in groups.items():
            sql = f"""
                SELECT mgyp, sequence
                FROM `{PROTEINS}`
                WHERE pfam_architecture LIKE @fam
                  AND biome LIKE @biome
                LIMIT @lim
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("fam", "STRING", f"%{fam}%"),
                        bigquery.ScalarQueryParameter("biome", "STRING", f"%{biome}%"),
                        bigquery.ScalarQueryParameter("lim", "INT64", per_bucket),
                    ]
                ),
            )
            print(f"[mgnify] {fam} / {gname} ({biome}) ...", flush=True)
            n = 0
            for row in job:
                records.append((row["mgyp"], row["sequence"], fam, gname))
                n += 1
            print(f"  -> {n} sequences", flush=True)
    return records


def parse_groups(spec):
    """'bacteria=2,archaea=2157' -> {'bacteria':'2','archaea':'2157'}"""
    out = {}
    for pair in spec.split(","):
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["uniprot", "curation", "mgnify"], default="uniprot")
    ap.add_argument("--outdir", required=True, help="Where to write seqs.fasta + metadata.tsv")
    ap.add_argument("--families", default="PF00069,PF00005,PF00072,PF07690,PF00271",
                    help="Comma-separated Pfam accessions (the shared label space)")
    ap.add_argument("--groups", default=None,
                    help="uniprot: name=taxid,... e.g. 'bacteria=2,archaea=2157'. "
                         "mgnify: name=biome_substring,... "
                         "Not required for --source curation (groups are fixed: "
                         "swissprot=reviewed, trembl=unreviewed).")
    ap.add_argument("--taxid", type=int, default=None,
                    help="curation only: optional taxonomy_id to hold fixed so the sole "
                         "shift is curation status (e.g. 2 = Bacteria).")
    ap.add_argument("--per_bucket", type=int, default=1000,
                    help="Max sequences per (family x group) bucket. "
                         "len(families)*len(groups)*per_bucket ~ total subset size.")
    ap.add_argument("--reviewed_only", action="store_true",
                    help="UniProt only: restrict to Swiss-Prot (cleaner, smaller).")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    if args.source == "curation":
        recs = fetch_uniprot_curation(families, args.per_bucket, taxid=args.taxid)
    elif args.source == "uniprot":
        if not args.groups:
            sys.exit("--groups is required for --source uniprot (e.g. 'bacteria=2,archaea=2157').")
        groups = {k: int(v) for k, v in parse_groups(args.groups).items()}
        recs = fetch_uniprot(families, groups, args.per_bucket, args.reviewed_only)
    else:  # mgnify
        if not args.groups:
            sys.exit("--groups is required for --source mgnify (name=biome_substring,...).")
        recs = fetch_mgnify(families, parse_groups(args.groups), args.per_bucket)

    if not recs:
        sys.exit("No sequences fetched -- check families/groups/credentials.")

    fasta_path = os.path.join(args.outdir, "seqs.fasta")
    meta_path = os.path.join(args.outdir, "metadata.tsv")
    with open(fasta_path, "w") as ff, open(meta_path, "w") as mf:
        mf.write("id\tfamily\tgroup\n")
        for acc, seq, fam, grp in recs:
            ff.write(f">{acc}\n{seq}\n")
            mf.write(f"{acc}\t{fam}\t{grp}\n")

    # quick summary
    from collections import Counter
    by_group = Counter(g for *_, g in recs)
    by_fam = Counter(f for _, _, f, _ in recs)
    print(f"\nWrote {len(recs)} sequences")
    print(f"  fasta: {fasta_path}")
    print(f"  meta:  {meta_path}")
    print(f"  groups:   {dict(by_group)}")
    print(f"  families: {dict(by_fam)}")
