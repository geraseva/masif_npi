import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from pathlib import Path
import Bio
from Bio.PDB import * 
from Bio.SeqUtils import IUPACData
from subprocess import Popen, PIPE
from multiprocessing import Pool
from pykeops.torch import LazyTensor

import os

from helper import *
from config import *

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

def find_modified_residues(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    with open(path, 'r') as f:
        for line in f:
            if line[:6] == 'SEQRES':
                for res in line.split()[4:]:
                    if res in PROTEIN_LETTERS:
                        continue
                    res_set.add(res)
            elif line[:4]=='ATOM':
                break
    return res_set


def load_structure_np(structure, chain_ids=None, 
                      modified=['LLP', 'KCX', 'FME', 'CSO', 'SEP', 'NH2', 'PCA', 'TPO', 'ACE', 'MSE']):
    
    if not isinstance(structure, Bio.PDB.Structure.Structure):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure, structure)
        modified = find_modified_residues(structure)

    coords = []
    types = []
    res=[]

    for chain in structure[0]:
        if (chain_ids == None) or (chain.get_id() in chain_ids):
            for residue in chain:
                het = residue.get_id()
                if (het[0] == " ") or (het[0][-3:] in modified):
                    for atom in residue:
                        coords.append(atom.get_coord())
                        types.append(atom.get_name())
                        res.append(residue.get_resname())
    coords = np.stack(coords)
    types_array = np.array(types)
    res=np.array(res)
       
    return {"atom_xyz": coords, "atom_types": types_array, "atom_resnames": res}

def download_pdb(pdb, dest, prot=True):
    # Download pdb 
    pdbl = PDBList()
    pdb_filename = pdbl.retrieve_pdb_file(pdb, pdir=tmp_dir, file_format='pdb')
    if prot:
        protonate(pdb_filename, dest)
    else:
        os.rename(pdb_filename, dest)

def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim",  in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", "-NOCon", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()

def encode_labels(labels,aa,onehot=0):

    d=aa.get('-')
    if d==None:
        d=0
    labels_enc=np.array([aa.get(a, d) for a in labels])
    if onehot>0:
        labels_enc=inttensor(labels_enc)
        labels_enc=F.one_hot(labels_enc,num_classes=onehot).float()
    else:
        labels_enc=tensor(labels_enc)
    return labels_enc

def encode_npy(p, encoders):
    
    if len(p)==0:
        return None
    list_to_onehot=['atom_types']

    protein_data={}

    protein_data['atom_xyz']=tensor(p['atom_xyz'])

    atom_types=[a[0] for a in p['atom_types']]
    for aa in encoders['atom_encoders']:
        o=max(aa['encoder'].values())+1 if aa['name'] in list_to_onehot else 0
        protein_data[aa['name']] = encode_labels(atom_types,aa['encoder'],o)
    
    if encoders.get('atomname_encoders') != None:
        for aa in encoders.get('atomname_encoders'):
            o=max(aa['encoder'].values())+1 if aa['name'] in list_to_onehot else 0
            protein_data[aa['name']] = encode_labels(p['atom_types'],aa['encoder'],o)

    if encoders.get('residue_encoders') != None:
        for la in encoders['residue_encoders']:
            o=max(la['encoder'].values())+1 if la['name'] in list_to_onehot else 0
            protein_data[la['name']] = encode_labels(p['atom_resnames'],la['encoder'],o)    

    mask=torch.ones(protein_data['atom_xyz'].shape[0], dtype=bool) # to mask H atoms, for example
    for key in protein_data:
        if 'mask' in key:
            mask=mask*protein_data.pop[key]
    mask=(mask>0)
    for key in protein_data:
        protein_data[key]=protein_data[key][mask]

    return protein_data

def load_protein_pair(filename, encoders, chains1, chains2=None):
    protein_pair = PairData()   
    parser = PDBParser(QUIET=True)
        
    try:
        structure = parser.get_structure('sample', filename)
        modified=find_modified_residues(filename)
            
        p1=load_structure_np(structure, chain_ids=chains1, modified=modified)
        p1 = encode_npy(p1, encoders=encoders)
        protein_pair.from_dict(p1, chain_idx=1)

        if chains2 is not None and chains2!='':
            p2=load_structure_np(structure, chain_ids=chains2, modified=modified)
            p2 = encode_npy(p2, encoders=encoders)
            protein_pair.from_dict(p2, chain_idx=2)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        protein_pair=None
    return protein_pair


class PairData(Data):
    def __init__(
        self,
        xyz_p1=None,
        xyz_p2=None,
        face_p1=None,
        face_p2=None,
        labels_p1=None,
        labels_p2=None,
        normals_p1=None,
        normals_p2=None,
        center_location_p1=None,
        center_location_p2=None,
        atom_xyz_p1=None,
        atom_xyz_p2=None,
        atom_types_p1=None,
        atom_types_p2=None,
        atom_center_p1=None,
        atom_center_p2=None,
        atom_res_p1=None,
        atom_res_p2=None,
        atom_rad_p1=None,
        atom_rad_p2=None,
        rand_rot_p1=None,
        rand_rot_p2=None,
        edge_labels_p1=None,
        edge_labels_p2=None  
          ):
        super().__init__()
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.face_p1 = face_p1
        self.face_p2 = face_p2
        self.labels_p1 = labels_p1
        self.labels_p2 = labels_p2
        self.edge_labels_p1=edge_labels_p1
        self.edge_labels_p2=edge_labels_p2
        self.normals_p1 = normals_p1
        self.normals_p2 = normals_p2
        self.center_location_p1 = center_location_p1
        self.center_location_p2 = center_location_p2
        self.atom_xyz_p1 = atom_xyz_p1
        self.atom_xyz_p2 = atom_xyz_p2
        self.atom_types_p1 = atom_types_p1
        self.atom_types_p2 = atom_types_p2
        self.atom_center_p1 = atom_center_p1
        self.atom_center_p2 = atom_center_p2
        self.atom_res_p1 = atom_res_p1
        self.atom_res_p2 = atom_res_p2
        self.atom_rad_p1 = atom_rad_p1
        self.atom_rad_p2 = atom_rad_p2
        self.rand_rot_p1 = rand_rot_p1
        self.rand_rot_p2 = rand_rot_p2

    def __inc__(self, key, value, *args, **kwargs):
        if ('face' in key) or ('edge' in key):
            if key[-3:]== '_p1':
                return self.xyz_p1.size(0)
            else:
                return self.xyz_p2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)
    
    def from_dict(self, mapping, chain_idx=None):
        if chain_idx==None:
            lbl=''
        else:   
            lbl=f'_p{chain_idx}'  

        for key in mapping:
            self.__setitem__(key+lbl,mapping[key])

    def to_dict(self, chain_idx=None):

        if chain_idx==None:
            return self._store.to_dict()
        else:
            P = {}
            lbl=f'_p{chain_idx}'
            for key in self.keys:
                if lbl in key:
                    P[key.replace(lbl,'')] = self.__getitem__(key)
            return P
        

class NpiDataset(Dataset):

    def __init__(self, root, list_file, encoders, prefix='',
        transform=None, pre_transform=None, pre_filter=None, 
        store=True):
        
        if isinstance(list_file,list):
            self.list = list_file
            self.name = prefix
        else:
            with open(list_file) as f_tr:
                self.list = f_tr.read().splitlines()
            self.name=prefix+list_file.split('/')[-1].split('.')[0]

        self.root=root
        self.raw_dir=root+'/raw/'
        self.processed_dir=root+'/processed/'

        self.transform = transform
        self.pre_transform=pre_transform
        self.pre_filter=pre_filter
        self.encoders=encoders
        
        if not all([os.path.exists(x) for x in self.processed_file_names]):
            for protonated_file in self.raw_file_names:
                if not os.path.exists(protonated_file):
                    download_pdb(protonated_file.split('/')[-1].split('.')[0],protonated_file)
            self.process()
        else:
            self.data = torch.load(self.processed_file_names[0], map_location='cuda')
            self.list = np.load(self.processed_file_names[1])

        if store:
            torch.save(self.data, self.processed_file_names[0])
            np.save(self.processed_file_names[1], self.list)
    
    @property
    def raw_file_names(self):
        file_names = [f'{self.raw_dir}/{x.split(" ")[0]}.pdb' for x in self.list]
        return file_names

    @property
    def processed_file_names(self):
        file_names = [
            self.processed_dir+self.name+'.pt',
            self.processed_dir+self.name+'_idx.npy'
        ]

        return file_names
    
    def load_single(self, idx):

        pspl=idx.split(' ')

        protein_pair=load_protein_pair(f'{self.raw_dir}/{pspl[0]}.pdb', 
                                       self.encoders, pspl[1], pspl[2] if len(pspl)==3 else None)
        if protein_pair is None:
            print(f'##! Skipping non-existing files for {idx}' )
        
        return protein_pair

    def process(self):
        
        print('# Loading pdb files', self.name)

        processed_dataset=[]
        processed_idx=[]

        #with Pool(4) as p:
        #    processed_dataset = list(tqdm(p.imap(self.load_single, self.list), total=len(self.list)))
        processed_dataset=[]
        for x in tqdm(self.list):
            processed_dataset.append(self.load_single(x))
        processed_idx=[idx for i, idx in enumerate(self.list) if processed_dataset[i]!=None]
        processed_dataset=[x for x in processed_dataset if x!=None]

        if self.pre_transform is not None:

            print('Preprocessing files', self.name)
            processed_dataset = [
                self.pre_transform(data) for data in tqdm(processed_dataset)
            ]

        if self.pre_filter is not None:
            processed_dataset = [data if self.pre_filter(data) else None for data in processed_dataset]
            processed_idx=[idx for i, idx in enumerate(processed_idx) if processed_dataset[i]!=None]
            processed_dataset=[x for x in processed_dataset if x!=None]            
        
        self.data=processed_dataset
        self.list=processed_idx

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        sample=self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class SurfacePrecompute(object):
    r"""Precomputation of surface"""

    def __init__(self, surf_gen, args):
        self.args=args
        self.preprocess_surface=surf_gen

    def __call__(self, protein_pair):

        P1 = {}
        P1["atom_xyz"] = protein_pair.atom_xyz_p1
        P1["atom_rad"] = protein_pair.atom_rad_p1
        if 'atom_xyz_p1_batch' in protein_pair.keys:
            P1['atom_xyz_batch']=protein_pair.atom_xyz_p1_batch
        else:
            P1['atom_xyz_batch']=torch.zeros(P1['atom_xyz'].shape[0], dtype=torch.int).to(P1['atom_xyz'].device)
        self.preprocess_surface(P1)
        protein_pair.xyz_p1 = P1["xyz"]
        protein_pair.normals_p1 = P1["normals"]
        protein_pair.face_p1 = P1.get("face")
        if 'atom_xyz_p1_batch' in protein_pair.keys:
            protein_pair.xyz_p1_batch=P1['xyz_batch']

        if not self.args.single_protein:
            P2 = {}
            P2["atom_xyz"] = protein_pair.atom_xyz_p2
            P2["atom_rad"] = protein_pair.atom_rad_p2
            if 'atom_xyz_p2_batch' in protein_pair.keys:
                P2['atom_xyz_batch']=protein_pair.atom_xyz_p2_batch
            else:
                P2['atom_xyz_batch']=torch.zeros(P2['atom_xyz'].shape[0], dtype=torch.int).to(P2['atom_xyz'].device)
            self.preprocess_surface(P2)      
            protein_pair.xyz_p2 = P2["xyz"]
            protein_pair.normals_p2 = P2["normals"]
            protein_pair.face_p2 = P2.get("face")
            if 'atom_xyz_p2_batch' in protein_pair.keys:
                protein_pair.xyz_p2_batch=P2['xyz_batch']
        return protein_pair


    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def get_threshold_labels(queries,batch_queries,source,batch_source,labels, threshold):

    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)
    
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1).detach()   # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1) < threshold**2
    )  
    
    query_labels = torch.take(labels,nn_i)
    query_labels=query_labels * nn_dist_i

    return query_labels


class LabelsFromAtoms(object):
    r"""Label surface points by complementary atom labels"""

    def __init__(self, threshold=5, single_protein=True):
        self.threshold=threshold
        self.single=single_protein

    def __call__(self, protein_pair):

        if protein_pair.atom_xyz_p2.shape[0]==0:
            query_labels=torch.zeros(protein_pair.xyz_p1.shape[0])
        else:
            query_labels=get_threshold_labels(
                queries = protein_pair.xyz_p1,
                batch_queries = None,
                source = protein_pair.atom_xyz_p2,
                batch_source = None,
                labels = protein_pair.atom_res_p2,
                threshold=self.threshold)
        protein_pair.labels_p1 = query_labels.detach()

        if not self.single:        
            if protein_pair.atom_xyz_p1.shape[0]==0:
                query_labels=torch.zeros(protein_pair.xyz_p2.shape[0])
            else:
                query_labels=get_threshold_labels(
                    queries = protein_pair.xyz_p2,
                    batch_queries = None,
                    source = protein_pair.atom_xyz_p1,
                    batch_source = None,
                    labels = protein_pair.atom_res_p1,
                    threshold=self.threshold)
            protein_pair.labels_p2 = query_labels.detach()

        return protein_pair


    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

class GenerateMatchingLabels(object):
    r"""For each receptor in pair find interacting surface points"""

    def __init__(self, threshold=2.0):

        self.threshold=threshold

    def __call__(self, protein_pair):

        xyz1_i = protein_pair.xyz_p1
        xyz2_j = protein_pair.xyz_p2

        xyz1_i = LazyTensor(xyz1_i[:, None, :].contiguous())
        xyz2_j = LazyTensor(xyz2_j[None, :, :].contiguous())

        xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1)
        xyz_dists = (self.threshold**2 - xyz_dists).step()

        protein_pair.labels_p1 = (xyz_dists.sum(1) > 1.0).float().view(-1).detach()
        protein_pair.labels_p2 = (xyz_dists.sum(0) > 1.0).float().view(-1).detach()

        pos_xyz1 = protein_pair.xyz_p1[protein_pair.labels_p1==1]
        pos_xyz2 = protein_pair.xyz_p2[protein_pair.labels_p2==1]

        pos_xyz_dists = (
            ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1)
        )
        edges=torch.nonzero(self.threshold**2 > pos_xyz_dists, as_tuple=True)

        protein_pair.edge_labels_p1=torch.nonzero(protein_pair.labels_p1)[edges[0]].view(-1).detach()
        protein_pair.edge_labels_p2=torch.nonzero(protein_pair.labels_p2)[edges[1]].view(-1).detach()

        return protein_pair

class RemoveSecondProtein(object):
    r"""Remove second protein information"""

    def __call__(self, protein_pair):

        for key in protein_pair.keys:
            if '_p2' in key:
                protein_pair.__delitem__(key)

        return protein_pair


class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __init__(self, as_single=False):

        self.as_single=as_single

    def __call__(self, data):

        R1 = torch.FloatTensor(Rotation.random().as_matrix()).to(data.xyz_p1.device)
        if self.as_single:
            R2=R1
            for key in data.keys: 
                if (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p1':
                    size=data.__getitem__(key).shape[0]
                    to_rotate=torch.cat([data.__getitem__(key),data.__getitem__(key[:-1]+'2')], dim=0)
                    to_rotate=torch.matmul(R1, to_rotate.T).T
                    data.__setitem__(key,to_rotate[:size])
                    data.__setitem__(key[:-1]+'2',to_rotate[size:])
        else:
            R2 = torch.FloatTensor(Rotation.random().as_matrix()).to(data.xyz_p1.device)
            for key in data.keys: 
                if (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p1':
                    data.__setitem__(key,torch.matmul(R1, data.__getitem__(key).T).T)
                elif (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p2':
                    data.__setitem__(key,torch.matmul(R2, data.__getitem__(key).T).T)
                    data.rand_rot2 = R2
                    
        data.rand_rot1 = R1
          
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CenterPairAtoms(object):
    r"""Centers a protein"""

    def __init__(self, as_single=False):

        self.as_single=as_single

    def __call__(self, data):
        
        if self.as_single:
            atom_center1=torch.cat(
                [data.atom_xyz_p1,data.atom_xyz_p2], dim=0
                ).mean(dim=0, keepdim=True)
            atom_center2=atom_center1
        else:
            atom_center1 = data.atom_xyz_p1.mean(dim=0, keepdim=True)
            try:
                atom_center2 = data.atom_xyz_p2.mean(dim=0, keepdim=True)
            except AttributeError:
                atom_center2=None

        data.atom_center1 = atom_center1

        for key in data.keys: 
            if ('xyz' in key) and key[-3:]=='_p1':
                data.__setitem__(key, data.__getitem__(key) - atom_center1)
            elif ('xyz' in key) and key[-3:]=='_p2':
                data.__setitem__(key, data.__getitem__(key) - atom_center2)
                data.atom_center2 = atom_center2

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def iface_valid_filter(protein_pair):
    labels1 = protein_pair.labels_p1.reshape(-1)>0
    valid1 = (
        (torch.sum(labels1) < 0.75 * len(labels1))
        and (torch.sum(labels1) > 30)
    )
    valid1 *= (labels1.shape[0]<30000)
    
    labels2 = protein_pair.get('labels_p2')
    if labels2 != None:
        labels2 = labels2.reshape(-1)>0
        valid2 = (
            (torch.sum(labels2) < 0.75 * len(labels2))
            and (torch.sum(labels2) > 30)
        )
        valid2 *= (labels2.shape[0]<30000)
    else:
        valid2=True

    return valid1 and valid2