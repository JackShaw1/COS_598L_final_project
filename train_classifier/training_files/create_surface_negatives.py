import csv
import gzip
import io
import os
import random
import tarfile
from Bio.PDB import MMCIFParser, ShrakeRupley

"""
This script gathers simulated small molecule binding pockets from the surfaces of randomly selected protein's from the AlphaFold Database's human proteome.
These simulated pockets are basically random, surface-exposed regions of proteins.
Since we also used control samples that were other small molecule binding pockets, we also needed to train our classifier to distinguish true Imatinib binding pockets from regions on proteins that were not binding pockets of any small molecule.
This script gathered these control sites to help our classifier make these dinstinctions.
"""

BASE_DIR = "train_classifier"
DEFAULT_TAR = "train_classifier/alphafold_human_proteome/human_proteome_AF_database.tar"
DEFAULT_OUT_DIR = "train_classifier/training_files/surface_negatives_apr21_2026"
DEFAULT_OUT_CSV = "train_classifier/training_files/surface_negatives.csv"

# Calculate average pLDDT for a residue
def residue_plddt(residue):
    values = []
    for atom in residue:
        values.append(atom.get_bfactor())
    return sum(values) / len(values)

# Extract uniprot id from filename
def parse_uniprot_id(member_name):
    name = os.path.basename(member_name)
    parts = name.split("-")
    return parts[1] if len(parts) >= 2 else name.replace(".cif.gz", "")

# Create a randomly-ordered list of residues that meet the SASA and pLDDT thresholds.
# This function uses the Shrake-Rupley method for calculating SASA values.
def find_surface_residues(structure, sasa_threshold, plddt_threshold, rng):
    ShrakeRupley().compute(structure, level="R")
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                sasa = getattr(residue, "sasa", 0)
                plddt = residue_plddt(residue)
                if sasa >= sasa_threshold and plddt >= plddt_threshold:
                    residues.append(residue)
        break
    # Randomly shuffle so we don't bias towards using residues from the N terminus of proteins
    rng.shuffle(residues)
    return residues

# Helper for streaming files from the .tar
def read_cif_from_tar(tf, member):
    f = tf.extractfile(member)
    if f is None:
        return None
    return gzip.decompress(f.read()).decode("utf-8", "replace")

def main():
    """
    Stream files from the tarball, retrieve candidate residues, save .cif files with candidate residues, and create output .csv
    """
    rng = random.Random(598)
    cif_parser = MMCIFParser(QUIET=True)
    rows = []
    examined = 0

    with tarfile.open(DEFAULT_TAR, "r:") as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and m.name.endswith(".cif.gz")
        ]

        rng.shuffle(members)

        for member in members:
            if len(rows) >= 100:
                break

            examined += 1
            uniprot_id = parse_uniprot_id(member.name)

            try:
                cif_text = read_cif_from_tar(tf, member)
                if cif_text is None:
                    continue

                structure = cif_parser.get_structure(
                    uniprot_id,
                    io.StringIO(cif_text),
                )

                residues = find_surface_residues(
                    structure,
                    sasa_threshold=20,
                    plddt_threshold=90,
                    rng=rng,
                )

                if not residues:
                    continue

                residue = residues[0]
                chain_id = residue.get_parent().get_id()
                resnum = residue.get_id()[1]

                cif_path = os.path.join(DEFAULT_OUT_DIR, f"{uniprot_id}.cif")

                with open(cif_path, "w") as f:
                    f.write(cif_text)

                rel_path = os.path.relpath(cif_path, BASE_DIR)
                rows.append([rel_path, chain_id, resnum])

                print(
                    f"[{len(rows):3d}/{100}] "
                    f"{uniprot_id} chain={chain_id} res={resnum}"
                )

            except Exception as e:
                print(f"Skipping {uniprot_id}: {e}")

    with open(DEFAULT_OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    main()