import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

"""
This script provides function for scoring using the classifier.
This includes:
 - Selecting surface-exposed residues
 - Builds an artificial binding pocket
 - Scores the pocket with the classifier
"""
RESIDUE_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
]

ELEMENT_TYPES = ["C", "N", "O", "S", "UNK"]

BACKBONE_ATOMS = {"N", "CA", "C", "O"}

RESIDUE_TO_IDX = {r: i for i, r in enumerate(RESIDUE_TYPES)}
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_TYPES)}

N_RESIDUE = len(RESIDUE_TYPES)
N_ELEMENT = len(ELEMENT_TYPES)
FEATURE_DIM = 3 + N_RESIDUE + N_ELEMENT

RADIUS = 5.0
MAX_ATOMS = 100

# Explicitly define the model architecture
class PocketClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutions
        self.conv1 = nn.Conv1d(29, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))

        # Global max pooling
        x = torch.max(x, dim=2).values

        x = self.dropout(functional.relu(self.fc1(x)))

        # Use sigmoid function to produce single score between 0 and 1
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)


def load_model(path, device):
    model = PocketClassifier().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# Choose surface exposed residues section
#
# Calculates average plddt for sample
def residue_plddt(residue) -> float | None:
    if "CA" in residue:
        return float(residue["CA"].get_bfactor())
    bfs = [a.get_bfactor() for a in residue.get_atoms()]
    if not bfs:
        return None
    return float(np.mean(bfs))

# Find residues that are surface-exposed based on the Shrake-Rupley method
def find_surface_residues(
    structure,
    max_residues,
    sasa_threshold,
    plddt_threshold,
    rng,
):
    sr = ShrakeRupley()
    # Calculates surface exposure at the residue level
    sr.compute(structure, level="R")

    candidates = []
    for struct_model in structure:
        for chain in struct_model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue

                sasa = getattr(residue, "sasa", None)
                if sasa is None or sasa < sasa_threshold:
                    continue

                plddt = residue_plddt(residue)
                if plddt is None or plddt < plddt_threshold:
                    continue

                candidates.append(residue)

        break

    if rng is None:
        rng = random.Random()

    if len(candidates) <= max_residues:
        rng.shuffle(candidates)
        return candidates

    return rng.sample(candidates, max_residues)


# Build synthetic binding pocket
# For classification
def extract_synthetic_pocket(target_residue, neighbor_search):
    virtual_ligand = []

    for atom in target_residue.get_atoms():
        name = atom.name.strip()

        if name in BACKBONE_ATOMS:
            continue

        elem = (atom.element or "").strip().upper()
        if elem == "H" or name.startswith("H"):
            continue

        virtual_ligand.append(atom)

    if not virtual_ligand:
        if "CA" in target_residue:
            virtual_ligand = [target_residue["CA"]]
        else:
            return None

    # Calculate centroid of residue so the artificial binding pocket is nornalized spatially
    centroid = np.mean(
        [a.get_vector().get_array() for a in virtual_ligand],
        axis=0,
    ).astype(np.float32)

    nearby = set()
    for la in virtual_ligand:
        nearby.update(neighbor_search.search(la.get_vector().get_array(), RADIUS))

    coords_list = []
    res_list = []
    elem_list = []

    for atom in nearby:
        residue = atom.get_parent()

        if residue is target_residue:
            continue

        if not is_aa(residue, standard=True):
            continue

        name = atom.name.strip()
        if name in BACKBONE_ATOMS:
            continue

        elem = (atom.element or "").strip().upper()
        if elem == "H" or name.startswith("H"):
            continue

        coord = (atom.get_vector().get_array() - centroid) / RADIUS

        resname = residue.resname.strip().upper()
        res_idx = RESIDUE_TO_IDX.get(resname, RESIDUE_TO_IDX["UNK"])
        elem_idx = ELEMENT_TO_IDX.get(elem, ELEMENT_TO_IDX["UNK"])

        coords_list.append(coord)
        res_list.append(res_idx)
        elem_list.append(elem_idx)

    if not coords_list:
        return None

    return (
        np.array(coords_list, dtype=np.float32),
        np.array(res_list, dtype=np.int64),
        np.array(elem_list, dtype=np.int64),
    )


# Create tensors for scoring
def pocket_to_tensor(
    coords,
    res_indices,
    elem_indices,
    rng):

    n = len(coords)

    res_oh = np.zeros((n, N_RESIDUE), dtype=np.float32)
    elem_oh = np.zeros((n, N_ELEMENT), dtype=np.float32)

    res_oh[np.arange(n), res_indices] = 1.0
    elem_oh[np.arange(n), elem_indices] = 1.0

    atom_feats = np.concatenate([coords, res_oh, elem_oh], axis=1)

    if n >= MAX_ATOMS:
        if rng is None:
            sel = np.random.choice(n, MAX_ATOMS, replace=False)
        else:
            sel = rng.choice(n, MAX_ATOMS, replace=False)

        atom_feats = atom_feats[sel]
    else:
        pad = np.zeros((MAX_ATOMS - n, FEATURE_DIM), dtype=np.float32)
        atom_feats = np.concatenate([atom_feats, pad], axis=0)

    return atom_feats.T


@torch.no_grad()
def score_pockets_batch(
    model,
    pockets,
    device,
    batch_size,
    rng):

    if not pockets:
        return np.zeros(0, dtype=np.float32)

    out = []

    for i in range(0, len(pockets), batch_size):
        batch = pockets[i:i + batch_size]
        tensors = np.stack([pocket_to_tensor(*p, rng=rng) for p in batch])
        x = torch.from_numpy(tensors).to(device)

        probs = model(x).detach().cpu().numpy()
        out.append(probs)

    return np.concatenate(out)