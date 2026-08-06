#!/usr/bin/env python
"""Fetch EVERY reviewed (Swiss-Prot) protein that carries an EC number.

Why this exists
---------------
`fetch_sequences.py` is Pfam-FIRST: it walks a hand-picked list of Pfam families
and pulls up to `per_bucket` proteins from each (family x taxon) cell. The
taxonomy ladder used 16 families x 5 domains, capped at 2500 per family, which is
why every EC analysis so far ran on 36,981 proteins -- 20,285 after requiring a
single unambiguous EC.

That cap is a sampling decision, not a data limit. Swiss-Prot holds ~280k
reviewed proteins with an EC number, and requiring a Pfam cross-reference costs
almost nothing (~277.7k still qualify). The binding constraint was the 16
families, not the Pfam requirement.

This script is EC-FIRST: one query, no family list, no per-family cap. It also
keeps the full NCBI lineage per protein so taxonomic groups can be defined
afterwards -- at phylum, class, or kingdom level -- without refetching. Group
assignment lives in `build_ec_dataset.py` for exactly that reason.

Output
------
  <outdir>/ec_swissprot_raw.tsv   id, ec, pfam, lineage_ids, organism_id, length
  <outdir>/ec_swissprot.fasta     one record per protein, id == accession

Both are written incrementally, so an interrupted run can be resumed with
--resume rather than restarted.
"""
import argparse
import os
import sys
import time

import requests

SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,ec,xref_pfam,lineage_ids,organism_id,length,sequence"


def total_results(query):
    """Ask UniProt how many hits there are before downloading any of them."""
    r = requests.get(SEARCH, params={"query": query, "size": 1, "format": "json"},
                     timeout=120)
    r.raise_for_status()
    return int(r.headers.get("x-total-results", 0))


def stream_pages(query, page_size, retries=5):
    """Yield parsed TSV rows, following UniProt's cursor pagination.

    UniProt paginates with a `Link: <...>; rel="next"` header rather than an
    offset, so the loop follows that URL instead of computing page numbers. A
    transient 5xx mid-download is retried in place: restarting a 280k-row pull
    from the beginning because of one blip is not acceptable.
    """
    params = {"query": query, "size": page_size, "format": "tsv", "fields": FIELDS}
    url = SEARCH
    header_seen = False
    while url:
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=300)
                if r.status_code == 200:
                    break
                # 429/5xx: back off and try the same cursor again
                print(f"  http {r.status_code}, retry {attempt+1}/{retries}", flush=True)
            except requests.RequestException as e:
                print(f"  {type(e).__name__}, retry {attempt+1}/{retries}", flush=True)
            time.sleep(2 ** attempt)
        else:
            raise SystemExit(f"giving up after {retries} retries at {url}")

        params = None  # the cursor URL already encodes them
        lines = r.text.split("\n")
        start = 0
        if lines and lines[0].startswith("Entry\t"):
            start = 1
            header_seen = True
        for line in lines[start:]:
            if line.strip():
                yield line.split("\t")
        url = r.links.get("next", {}).get("url")
    if not header_seen:
        print("  WARNING: never saw a TSV header -- field list may be wrong", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--query", default="(reviewed:true) AND (ec:*)",
                    help="UniProt query. Default = all Swiss-Prot with any EC number.")
    ap.add_argument("--page_size", type=int, default=500,
                    help="rows per request; 500 is UniProt's max for TSV")
    ap.add_argument("--max_len", type=int, default=1022,
                    help="drop sequences longer than this (ESM-C context); 0 = keep all")
    ap.add_argument("--resume", action="store_true",
                    help="skip accessions already present in the output TSV")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tsv_path = os.path.join(args.outdir, "ec_swissprot_raw.tsv")
    fa_path = os.path.join(args.outdir, "ec_swissprot.fasta")

    seen = set()
    if args.resume and os.path.exists(tsv_path):
        with open(tsv_path) as f:
            f.readline()
            for line in f:
                seen.add(line.split("\t", 1)[0])
        print(f"resume: {len(seen)} accessions already downloaded", flush=True)

    n_total = total_results(args.query)
    print(f"query : {args.query}")
    print(f"hits  : {n_total}", flush=True)

    mode = "a" if (args.resume and seen) else "w"
    n_kept = n_long = 0
    with open(tsv_path, mode) as tf, open(fa_path, mode) as ff:
        if mode == "w":
            tf.write("id\tec\tpfam\tlineage_ids\torganism_id\tlength\n")
        for row in stream_pages(args.query, args.page_size):
            if len(row) < 7:
                continue
            acc, ec, pfam, lineage, org, length, seq = row[:7]
            if acc in seen:
                continue
            if args.max_len and len(seq) > args.max_len:
                n_long += 1
                continue
            seen.add(acc)
            tf.write(f"{acc}\t{ec}\t{pfam}\t{lineage}\t{org}\t{length}\n")
            ff.write(f">{acc}\n{seq}\n")
            n_kept += 1
            if n_kept % 10000 == 0:
                tf.flush(); ff.flush()
                print(f"  {n_kept} kept ({n_long} too long) of ~{n_total}", flush=True)

    print(f"\nWrote {n_kept} proteins ({n_long} dropped for length > {args.max_len})")
    print(f"  {tsv_path}")
    print(f"  {fa_path}")
