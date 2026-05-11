import csv
import random
from Bio.PDB import MMCIFParser
import requests

"""
This script generates negative training samples for our classifier.
These negative training samples are binding pockets of other small molecules (not Imatinib).
The names of the small molecules used to create the negative training set are in train_classifier/training_files/build_controls/pdb_id_molecules_search.txt.
"""

# PDB entries that contain at least one protein and at least one of the following small molecules...
# ATP, ADP, GTP, GDP, GLC, NAD, NAP, FAD, FMN, COA, PYR, LAC, and CIT 
PDB_ID_FILE = "train_classifier/training_files/build_controls/pdb_ids.txt"
# List of ligand ids with full names
LIGAND_FILE = "train_classifier/training_files/build_controls/pdb_id_molecules_search.txt"
OUTPUT_DIR = "train_classifier/training_files/unique_control_structures_apr19_2026"
OUTPUT_CSV = "train_classifier/training_files/negatives.csv"
# Url for downloading corresponding .cif file from PDB
MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

parser = MMCIFParser(QUIET=True)

def load_pdb_ids(path):
    pdb_ids = []
    with open(path, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            for value in row:
                pdb_id = value.strip().upper()
                if pdb_id:
                    pdb_ids.append(pdb_id)
    return pdb_ids

def load_target_ligands(path):
    ligands = set()
    with open(path, "r") as file:
        for line in file:
            if line.startswith("- "):
                ligand = line.strip()[2:].split()[0].upper()
                ligands.add(ligand)
    return ligands

def download_mmcif(pdb_id):
    output_path = f"{OUTPUT_DIR}/{pdb_id.lower()}.cif"
    url = MMCIF_URL.format(pdb_id=pdb_id)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        file.write(response.content)
    return output_path

def find_first_target_ligand(structure, target_ligands):
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_name = residue.get_resname().strip().upper()
                if residue_name in target_ligands:
                    chain_id = chain.get_id()
                    residue_number = residue.get_id()[1]
                    return chain_id, residue_number
        break
    return None

def main():

    # Load and parse PDB ids, which are all on the same line
    pdb_ids = load_pdb_ids(PDB_ID_FILE)
    # Loads ids of target small molecules
    target_ligands = load_target_ligands(LIGAND_FILE)

    random.Random(598).shuffle(pdb_ids)

    print(f"Loaded {len(pdb_ids)} PDB IDs")
    print(f"Loaded {len(target_ligands)} target ligands")

    with open(OUTPUT_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        for i, pdb_id in enumerate(pdb_ids, start=1):
            print(f"[{i}/{len(pdb_ids)}] Processing {pdb_id}", flush=True)
            try:
                cif_path = download_mmcif(pdb_id)
            except:
                continue
            structure = parser.get_structure(pdb_id, str(cif_path))
            ligand_hit = find_first_target_ligand(structure, target_ligands)
            if ligand_hit is None:
                continue
            chain_id, residue_number = ligand_hit
            row = (str(cif_path), chain_id, str(residue_number))
            writer.writerow(row)

if __name__ == "__main__":
    main()