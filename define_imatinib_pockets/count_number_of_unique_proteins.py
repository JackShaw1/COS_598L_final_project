import os
import shutil

"""
This script identifies the set of unique proteins from the PDB that are in complex with Imatinib.
The find repeated proteins, this script parses the PDB files' headers to identify the UniProt code(s) present in each PDB id.
From here, the script counts the number of original files that contain a unique protein.
These PDB ids were originally retrieved from the PDB using an advanced to search to find deposited structure with the ligand STI (Imatinib) and at least one protein.
The .pdb files were then downloaded from the PDB, and analyzed with this script as well as others.
"""

INPUT_DIR = "define_imatinib_pockets/imatinib_structures_apr15_2026"
OUTPUT_DIR = "define_imatinib_pockets/unique_imatinib_structures_apr15_2026"

# Parse PDB and retrieve UniProt accessions from the header
def get_uniprot_accessions(pdb_path):
    accessions = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("DBREF"):
                continue

            fields = line.split()
            if "UNP" in fields:
                accessions.add(fields[fields.index("UNP") + 1])

    return accessions

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_uniprots = set()
    kept_count = 0

    pdb_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".pdb"))

    for pdb_file in pdb_files:

        # List all files and retrieve UniProt accessions from file headers
        pdb_path = os.path.join(INPUT_DIR, pdb_file)
        uniprots = get_uniprot_accessions(pdb_path)

        # Print when skipping a file due to repeated UniProt code
        if seen_uniprots.intersection(uniprots):
            print(f"Skipping {pdb_file}: already saw {sorted(uniprots)}")
            continue

        # Copy original file to deduplicated folder
        shutil.copy2(pdb_path, os.path.join(OUTPUT_DIR, pdb_file))
        seen_uniprots.update(uniprots)
        kept_count += 1

    # Print total number kept
    print(f"\nKept {kept_count} unique structures in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
