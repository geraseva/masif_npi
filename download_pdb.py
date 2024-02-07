import Bio
from Bio.PDB import * 
from Bio.SeqUtils import IUPACData
import sys
import importlib
import os
import numpy as np
from subprocess import Popen, PIPE
from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *


parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument(
    "--pdb", type=str,default='', help="PDB code along with chains to extract, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--pdb_list", type=str,default='', help="Path to a text file that includes a list of PDB codes along with chains, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--save_npys", type=bool ,default=False, help="Whether to save atom coordinates and names as npy files")



tmp_dir = Path('/home/domain/tmp')
pdb_dir = Path('/home/domain/data/geraseva/masif/data/masif_npi/data_preparation/00-raw_pdbs')
npy_dir = Path('./npi_dataset/raw')

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1" 


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set

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
        coords.append(atom.get_coord())
        types.append(atom.get_name())
        res.append(atom.get_parent().get_resname())

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
        np.save(npy_dir / (p.stem + "_resnames.npy"), protein["resnames"])       

def extractPDB(
    infilename, outfilename, chain_ids=None 
):
    # extract the chain_ids from infilename and save in outfilename. 
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()



def get_single(pdb_id: str,chains: list, prot=True):
    protonated_file = pdb_dir/f"{pdb_id}.pdb"
    if not protonated_file.exists():
        # Download pdb 
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir,file_format='pdb')

        if prot:
            protonate(pdb_filename, protonated_file)

    pdb_filename = protonated_file

    # Extract chains of interest.
    p={}
    for chain in chains:
        out_filename = pdb_dir/f"{pdb_id}_{chain}.pdb"
        try:
            extractPDB(pdb_filename, str(out_filename), chain)
            protein = load_structure_np(out_filename,center=False)
        except:
            print('Fail to extract', pdb_id, chain)
        p[chain]=protein
    return p


if __name__ == '__main__':
    args = parser.parse_args()
    if args.pdb != '':
        pdb_id = args.pdb.split('_')
        chains = pdb_id[1:]
        pdb_id = pdb_id[0]
        get_single(pdb_id,chains)

    elif args.pdb_list != '':
        with open(args.pdb_list) as f:
            pdb_list = f.read().splitlines()
        for pdb_id in tqdm(pdb_list):
           pdb_id = pdb_id.split('_')
           chains = pdb_id[1:]
           pdb_id = pdb_id[0]
           p=get_single(pdb_id,chains, args.save_res)
           if args.save_npys:
               for chain in p:
                   np.save(npy_dir / f"{pdb_id}_{chain}_atomxyz", p[chain]["xyz"])
                   np.save(npy_dir / f"{pdb_id}_{chain}_atomtypes", p[chain]["types"])
                   np.save(npy_dir / f"{pdb_id}_{chain}_resnames", p[chain]["resnames"])
               
    else:
        raise ValueError('Must specify PDB or PDB list') 