import os
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import Counter

"""
This script creates Figure 1b, which shows the frequency of amino acid contacts within 5 Å of Imatinib across all unique interactors.
We used Biopython.PDB to take these measurements, which makes parsing of structure files (.pdb) very straightforward.
"""

aa_codes = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]

parser = PDBParser(QUIET=True)
bind_dist = []

# Nested for loop to collect amino acid contacts within 5 Å of imatinib
for file in os.listdir("define_imatinib_pockets/unique_imatinib_structures_apr15_2026"):
    if not file.endswith(".pdb"):
        continue
    found_neighbors = []
    structure = parser.get_structure('struct', f"define_imatinib_pockets/unique_imatinib_structures_apr15_2026/{file}")
    searcher = NeighborSearch(list(structure.get_atoms()))
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "STI":
                    for atom in residue:
                        # Neighbor searching to collect all atoms within 5 Angstroms of an atom from the Imatinib molecules
                        neighbors = searcher.search(atom.get_coord(), 5.0)
                        for neighbor in neighbors:
                            if neighbor.get_parent().get_resname() in aa_codes:
                                neighbor_id = f"{neighbor.get_parent().get_parent().get_id()} - {neighbor.get_parent().get_id()[1]}"
                                if neighbor_id not in found_neighbors:
                                    found_neighbors.append(neighbor_id)
                                    bind_dist.append(neighbor.get_parent().get_resname())
        break

# Generate amino acid counts
counts = Counter(bind_dist)
frequencies = {aa: counts.get(aa, 0) for aa in aa_codes}
frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))

uniform_y = sum(frequencies.values()) / len(frequencies)

# Plot
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(frequencies.keys(), frequencies.values(), color='steelblue', edgecolor='black', linewidth=0.5)
ax.axhline(uniform_y, color="pink", linestyle="--", linewidth=3, label=f"Uniform = {uniform_y:.2f}")
ax.set_xlabel("Amino Acid", fontsize=12)
ax.set_ylabel("Contact Frequency", fontsize=12)
ax.tick_params(axis='x', labelrotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.savefig("define_imatinib_pockets/Figure_1b.png", dpi=300)