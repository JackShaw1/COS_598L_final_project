import csv
from Bio.PDB import PDBParser, NeighborSearch
import os

"""
Create positive training set of Imatinib binding pockets (deduplicated) from PDB.
This script collects the chain id and residue indices of each Imatinib binding pocket in the PDB that is unique.
"""

parser = PDBParser(QUIET=True)

for file in os.listdir("define_imatinib_pockets/unique_imatinib_structures_apr15_2026"):
    structure = parser.get_structure('struct', f"define_imatinib_pockets/unique_imatinib_structures_apr15_2026/{file}")
    identity = False
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "STI" and identity == False:
                    with open("train_classifier/training_files/positives.csv", "a", newline="") as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow([f"../define_imatinib_pockets/unique_imatinib_structures_apr15_2026/{file}", chain.get_id(), residue.get_id()[1]])
                    identity = True
        break