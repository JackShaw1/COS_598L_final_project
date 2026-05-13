import csv
import gzip
import os
import random
import tarfile
import time
import warnings
import numpy as np
import torch
from Bio.PDB import NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser
from pocket_scoring import (
    extract_synthetic_pocket,
    find_surface_residues,
    load_model,
    residue_plddt,
    score_pockets_batch,
)

"""
This script scans the AlphaFold Database's human proteome for putative imatinib binding pockets.

Workflow:
  - Open the tar archive and randomly shuffle the .cif.gz members
  - Parse each selected structure with MMCIFParser, run Shrake-Rupley,
    randomly sample up to MAX_RESIDUES surface-exposed residues
  - For each sampled residue build a synthetic pocket and score it with
    the trained classifier
  - Append every pocket whose probability >= SCORE_THRESHOLD to the output CSV
"""

warnings.filterwarnings("ignore")

TAR_PATH = "train_classifier/alphafold_human_proteome/human_proteome_AF_database.tar"
MODEL_PATH = "train_classifier/pocket_classifier.pt"
OUTPUT_CSV = "scan_proteome/imatinib_pocket_hits.csv"

MAX_RESIDUES = 15
SASA_THRESHOLD = 20.0
PLDDT_THRESHOLD = 80.0
SCORE_THRESHOLD = 0.5
BATCH_SIZE = 4
SEED = 598
DEVICE = "cpu"
PROGRESS_EVERY = 50


def parse_uniprot_id(member_name):
    """AF-A0A024R1R8-F1-model_v6.cif.gz → 'A0A024R1R8'."""
    base = os.path.basename(member_name)
    parts = base.split("-")
    return parts[1] if len(parts) >= 2 else base

def make_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    device = "cpu"

    make_parent_dir(OUTPUT_CSV)

    model = load_model(MODEL_PATH, device)
    cif_parser = MMCIFParser(QUIET=True)

    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    out_fh = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(out_fh)

    writer.writerow([
        "uniprot_id",
        "chain_id",
        "residue_number",
        "residue_name",
        "sasa",
        "plddt",
        "score",
    ])
    out_fh.flush()

    n_proteins = 0
    n_hits_total = 0
    # Track time to estimate runtime end
    t0 = time.time()

    try:
        # Load AlphaFold Database's human proteome
        with tarfile.open(TAR_PATH, mode="r") as tf:
            members = [
                member
                for member in tf.getmembers()
                if member.isfile() and member.name.endswith(".cif.gz")
            ]

            # Shuffle files randomly so they aren't processed alphabetically
            py_rng.shuffle(members)

            for member in members:
                uniprot_id = parse_uniprot_id(member.name)

                try:
                    fobj = tf.extractfile(member)

                    if fobj is None:
                        continue

                    with gzip.open(fobj, "rt", encoding="utf-8", errors="replace") as cif_fh:
                        structure = cif_parser.get_structure(uniprot_id, cif_fh)

                except Exception as exc:
                    print(f"[{uniprot_id}] parse error: {exc}")
                    continue

                try:
                    surface = find_surface_residues(
                        structure,
                        max_residues=MAX_RESIDUES,
                        sasa_threshold=SASA_THRESHOLD,
                        plddt_threshold=PLDDT_THRESHOLD,
                        rng=py_rng,
                    )

                except Exception as exc:
                    print(f"[{uniprot_id}] SASA error: {exc}")
                    continue

                ns = NeighborSearch(list(structure.get_atoms()))

                pockets = []
                meta = []

                for residue in surface:
                    feat = extract_synthetic_pocket(residue, ns)

                    if feat is None:
                        continue

                    pockets.append(feat)

                    meta.append((
                        residue.get_parent().id,
                        int(residue.id[1]),
                        residue.resname.strip().upper(),
                        float(getattr(residue, "sasa", 0.0)),
                        residue_plddt(residue) or 0.0,
                    ))

                if pockets:
                    probs = score_pockets_batch(
                        model,
                        pockets,
                        device,
                        batch_size=BATCH_SIZE,
                        rng=np_rng,
                    )

                    n_hits = 0
                    max_p = float(probs.max())

                    for (chain_id, resnum, resname, sasa, plddt), prob in zip(meta, probs):
                        if prob >= SCORE_THRESHOLD:
                            writer.writerow([
                                uniprot_id,
                                chain_id,
                                resnum,
                                resname,
                                f"{sasa:.2f}",
                                f"{plddt:.2f}",
                                f"{prob:.4f}",
                            ])

                            n_hits += 1

                    out_fh.flush()
                    n_hits_total += n_hits

                else:
                    n_hits = 0
                    pass

                n_proteins += 1

                if n_proteins % PROGRESS_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = n_proteins / elapsed if elapsed > 0 else 0.0

                    print(
                        f"[progress] {n_proteins} proteins  "
                        f"{n_hits_total} hits  "
                        f"{rate:.2f} prot/s  "
                        f"{elapsed / 60:.1f} min elapsed"
                    )

    finally:
        out_fh.close()

    print("complete")


if __name__ == "__main__":
    main()