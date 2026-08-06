#!/usr/bin/env python
"""Annotate an existing sequence set with EC numbers from UniProt.

Why this exists
---------------
Every measurement so far (measure_real_geometry.py, the alpha/beta calibration)
uses Pfam FAMILY as the label space. Families are defined by homology, so a
"family shift" and a "sequence-similarity shift" are close to the same thing --
which makes it impossible to tell whether a taxonomic shift damages FUNCTION or
merely moves sequences around.

EC number gives a second, functionally-defined label axis over the SAME proteins:
the same reaction is catalysed in bacteria and in plants, and often by more than
one Pfam family. That lets us ask the two questions the family label cannot:

  * within one EC, what does the bacteria -> plants shift look like?
  * within one EC, what does the family_1 -> family_2 shift look like?

Reads  <raw>/metadata.tsv   (id, family, group  -- written by fetch_sequences.py)
Writes <raw>/ec_annotations.tsv  (id, ec, protein_name, organism)

`ec` is UniProt's "EC number" field verbatim: possibly empty (not an enzyme /
not annotated), possibly several ECs separated by "; ", and possibly partial
("2.7.-.-"). Downstream code decides how to handle those; nothing is dropped here.

Resumable: rerunning skips accessions already present in the output file, so an
interrupted run costs only the batches it had not reached.
"""
import argparse
import os
import sys
import time

import requests

ENDPOINT = "https://rest.uniprot.org/uniprotkb/accessions"
FIELDS = "accession,ec,protein_name,organism_name"


def read_ids(meta_path):
    ids = []
    with open(meta_path) as f:
        f.readline()  # header: id\tfamily\tgroup
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                ids.append(parts[0])
    return ids


def read_done(out_path):
    """Accessions already fetched, so a rerun resumes instead of restarting."""
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path) as f:
        f.readline()
        for line in f:
            acc = line.split("\t", 1)[0].strip()
            if acc:
                done.add(acc)
    return done


def fetch_batch(accessions, retries=4, timeout=60):
    """One UniProt call for up to `batch_size` accessions -> list of TSV rows.

    UniProt returns NO row for an accession it cannot resolve (deleted, demerged),
    so the caller must not assume input and output line up one-to-one.
    """
    params = {"accessions": ",".join(accessions), "fields": FIELDS, "format": "tsv"}
    for attempt in range(retries):
        try:
            r = requests.get(ENDPOINT, params=params, timeout=timeout)
            if r.status_code == 200:
                lines = r.text.rstrip("\n").split("\n")
                return [ln for ln in lines[1:] if ln.strip()]  # drop header
            # 429/5xx are transient; back off and retry
            print(f"    http {r.status_code}: {r.text[:150]}", flush=True)
        except requests.RequestException as e:
            print(f"    request error: {e}", flush=True)
        time.sleep(2 ** attempt)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, help="metadata.tsv with an `id` column")
    ap.add_argument("--out", required=True, help="output ec_annotations.tsv")
    ap.add_argument("--batch_size", type=int, default=100,
                    help="accessions per UniProt call (100 is the documented max)")
    ap.add_argument("--sleep", type=float, default=0.15, help="pause between calls")
    args = ap.parse_args()

    ids = read_ids(args.meta)
    done = read_done(args.out)
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} accessions in {args.meta}; {len(done)} already fetched; "
          f"{len(todo)} to go")
    if not todo:
        sys.exit(0)

    new_file = not os.path.exists(args.out)
    n_rows = 0
    with open(args.out, "a") as out:
        if new_file:
            out.write("id\tec\tprotein_name\torganism\n")
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            rows = fetch_batch(batch)
            if rows is None:
                print(f"  giving up on batch at offset {start} after retries; "
                      f"rerun to retry it", flush=True)
                continue
            for row in rows:
                out.write(row.rstrip("\n") + "\n")
                n_rows += 1
            out.flush()
            done_n = min(start + args.batch_size, len(todo))
            print(f"  {done_n}/{len(todo)} requested, {n_rows} rows written", flush=True)
            time.sleep(args.sleep)

    print(f"\nWrote {n_rows} rows -> {args.out}")
