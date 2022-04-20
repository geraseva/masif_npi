import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    res=[]
    for atom in atoms:
        if atom.element in ele2num.keys():
            coords.append(atom.get_coord())
            types.append(atom.element)
            res.append(atom.get_parent().get_resname())
        else: 
            print('Unknown atom',atom.element, 'in', str(fname).split('/')[-1])

    coords = np.stack(coords)
    types_array = np.array(types)
    res=np.array(res)

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
        
    return {"xyz": coords, "types": types_array, "resnames": res}


def convert_pdbs(pdb_dir, npy_dir):
    print("Converting PDBs")
    for p in tqdm(pdb_dir.glob("*.pdb")):
        protein = load_structure_np(p, center=False)
        np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])
