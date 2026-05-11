import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser

"""
This script is responsible for gathering structural information for building input samples as well as creating augmented versions of these samples.
This script creates transformed versions of the original samples (to get five transformations for each original sample).
These transformations consist of small perturbations to each original atom coordinates, following by random three dimensional rotations.
The purpose of these transformations are to make our classifier robust to noise in AlphaFold predictions as well as arbitrary rotations in the XYZ coordinate plane that do not help to identify true Imatinib binding pockets.
"""

RESIDUE_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "UNK"
]

ELEMENT_TYPES = ["C", "N", "O", "S", "UNK"]
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Converts amino acid names to numbers for one-hot encoding handling
RESIDUE_TO_INDEX = {residue: i for i, residue in enumerate(RESIDUE_TYPES)}
ELEMENT_TO_INDEX = {element: i for i, element in enumerate(ELEMENT_TYPES)}

N_RESIDUE = 21
N_ELEMENT = 5
FEATURE_DIM = 29

RADIUS = 5.0
MAX_ATOMS = 100

# Loads the file as a Biopython.PDB structure object
def load_structure(path):
    if path.endswith(".pdb"):
        return PDBParser(QUIET=True).get_structure("", path)
    return MMCIFParser(QUIET=True).get_structure("", path)

# Finds Biopython.PDB residue object
def find_residue(structure, chain_id, residue_number):
    residue_number = int(residue_number)
    for model in structure:
        for chain in model:
            if chain.get_id() != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] == residue_number:
                    return residue
        return None

# Gather additional atomic information
def atom_sort_key(atom):
    residue = atom.get_parent()
    chain = residue.get_parent()
    return (
        chain.get_id(),
        residue.get_id()[1],
        residue.get_id()[2],
        atom.get_name().strip(),
    )

# Collect all atoms within 5 Angstroms of at least one atom from the ligand or simulated ligand pocket
def atoms_near_ligand(structure, ligand_atoms):
    searcher = NeighborSearch(list(structure.get_atoms()))
    nearby_atoms = set()
    for atom in ligand_atoms:
        nearby_atoms.update(searcher.search(atom.get_coord(), RADIUS))
    return sorted(nearby_atoms, key=atom_sort_key)

# Create feature arrays for each atom
def make_feature_arrays(atoms, center, residue_to_exclude):
    coords = []
    residue_indices = []
    element_indices = []

    for atom in atoms:

        # Don't include backbones atoms or hydrogens
        if atom.get_name() in ['N', 'CA', 'C', 'O', 'H']:
            continue
        if 'N' not in atom.get_name() and 'C' not in atom.get_name() and 'O' not in atom.get_name() and 'S' not in atom.get_name():
            continue

        # If the residue is the ligand itself, don't include it
        residue = atom.get_parent()
        if residue is residue_to_exclude:
            continue
        residue_name = residue.get_resname().strip().upper()
        element = (atom.element or "").strip().upper()

        # Normalize atom coordinates by dividing by 5 Angstroms (the maximum distance for an atom to be considered)
        coords.append((atom.get_coord() - center) / RADIUS)
        residue_indices.append(RESIDUE_TO_INDEX.get(residue_name, RESIDUE_TO_INDEX["UNK"]))
        element_indices.append(ELEMENT_TO_INDEX.get(element, ELEMENT_TO_INDEX["UNK"]))

    if not coords:
        return None

    return (
        np.array(coords, dtype=np.float32),
        np.array(residue_indices, dtype=np.int64),
        np.array(element_indices, dtype=np.int64),
    )

# Generate random rotation matrix
def random_rotation_matrix(rng):
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)

# Apply random rotational matrix to the coordinates only
def random_rotate_sample(sample, rng):
    coords, residue_indices, element_indices = sample
    rotation = random_rotation_matrix(rng)
    rotated_coords = coords @ rotation.T
    return (
        rotated_coords.astype(np.float32),
        residue_indices.copy(),
        element_indices.copy(),
    )

# For each atom in the residue (ligand), gather all nearby atoms, normalize them so the centroid coordinates of the residue (ligand), and then divide them by 5
def extract_pocket_features(path, chain_id, residue_number):
    structure = load_structure(path)
    ligand = find_residue(structure, chain_id, residue_number)
    ligand_atoms = list(ligand.get_atoms())
    center = np.mean([atom.get_coord() for atom in ligand_atoms], axis=0)
    nearby_atoms = atoms_near_ligand(structure, ligand_atoms)
    features = make_feature_arrays(nearby_atoms, center, ligand)
    return features

# Function for handling simulation small molecule binding pockets by identifying random residues with good surface exposure
def extract_surface_pocket_features(path, chain_id, residue_number):
    structure = load_structure(path)
    surface_residue = find_residue(structure, chain_id, residue_number)
    virtual_ligand_atoms = []
    for atom in surface_residue:
        virtual_ligand_atoms.append(atom)
    # Calculate centroid of simulated binding molecule
    center = np.mean([atom.get_coord() for atom in virtual_ligand_atoms], axis=0)
    nearby_atoms = atoms_near_ligand(structure, virtual_ligand_atoms)
    features = make_feature_arrays(nearby_atoms, center, surface_residue)
    return features