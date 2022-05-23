import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.transforms import Compose
import numpy as np
from scipy.spatial.transform import Rotation
import math
import urllib.request
import tarfile
from pathlib import Path
import requests
from data_preprocessing.convert_pdb2npy import convert_pdbs
from data_preprocessing.convert_ply2npy import convert_plys
from data_iteration import project_npi_labels
from tqdm import tqdm

tensor = torch.FloatTensor
inttensor = torch.LongTensor


def numpy(x):
    return x.detach().cpu().numpy()


def iface_valid_filter(protein_pair):
    labels1 = protein_pair.gen_labels_p1.reshape(-1)>0
    valid1 = (
        (torch.sum(labels1) < 0.75 * len(labels1))
        and (torch.sum(labels1) > 30)
    )
    
    labels2 = protein_pair.get('gen_labels_p2')
    if labels2 != None:
        labels2 = labels2.reshape(-1)>0
        valid2 = (
            (torch.sum(labels2) < 0.75 * len(labels2))
            and (torch.sum(labels2) > 30)
            and (torch.sum(labels2) > 0.01 * labels1.shape[0])
        )
        valid1 = (
            valid1 
            and (torch.sum(labels1) > 0.01 * labels2.shape[0])
        )
    else:
        valid2=True

    return valid1 and valid2



class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __call__(self, data):
        R1 = tensor(Rotation.random().as_matrix())
        R2 = tensor(Rotation.random().as_matrix())

        data.rand_rot1 = R1
        data.rand_rot2 = R2

        data.atom_coords_p1 = torch.matmul(R1, data.atom_coords_p1.T).T
        data.atom_coords_p2 = torch.matmul(R2, data.atom_coords_p2.T).T

        try:
            data.xyz_p1 = torch.matmul(R1, data.xyz_p1.T).T
            data.normals_p1 = torch.matmul(R1, data.normals_p1.T).T

            data.xyz_p2 = torch.matmul(R2, data.xyz_p2.T).T
            data.normals_p2 = torch.matmul(R2, data.normals_p2.T).T

        except AttributeError:
            return data 


        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CenterPairAtoms(object):
    r"""Centers a protein"""

    def __call__(self, data):
        atom_center1 = data.atom_coords_p1.mean(dim=-2, keepdim=True)
        atom_center2 = data.atom_coords_p2.mean(dim=-2, keepdim=True)

        data.atom_coords_p1 = data.atom_coords_p1 - atom_center1
        data.atom_coords_p2 = data.atom_coords_p2 - atom_center2

        data.atom_center1 = atom_center1
        data.atom_center2 = atom_center2

        try:
            data.xyz_p1 = data.xyz_p1 - atom_center1
            data.xyz_p2 = data.xyz_p2 - atom_center2
        except AttributeError:
            return data 

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeChemFeatures(object):
    r"""Centers a protein"""

    def __call__(self, data):
        pb_upper = 3.0
        pb_lower = -3.0

        try:
            chem_p1 = data.chemical_features_p1
            chem_p2 = data.chemical_features_p2
        except AttributeError:
            return data

        pb_p1 = chem_p1[:, 0]
        pb_p2 = chem_p2[:, 0]
        hb_p1 = chem_p1[:, 1]
        hb_p2 = chem_p2[:, 1]
        hp_p1 = chem_p1[:, 2]
        hp_p2 = chem_p2[:, 2]

        # Normalize PB
        pb_p1 = torch.clamp(pb_p1, pb_lower, pb_upper)
        pb_p1 = (pb_p1 - pb_lower) / (pb_upper - pb_lower)
        pb_p1 = 2 * pb_p1 - 1

        pb_p2 = torch.clamp(pb_p2, pb_lower, pb_upper)
        pb_p2 = (pb_p2 - pb_lower) / (pb_upper - pb_lower)
        pb_p2 = 2 * pb_p2 - 1

        # Normalize HP
        hp_p1 = hp_p1 / 4.5
        hp_p2 = hp_p2 / 4.5

        data.chemical_features_p1 = torch.stack([pb_p1, hb_p1, hp_p1]).T
        data.chemical_features_p2 = torch.stack([pb_p2, hb_p2, hp_p2]).T

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def load_protein_npy(pdb_id, data_dir, center=False, single_pdb=False, atom_encoder=None,label_encoder=None):
    """Loads a protein surface mesh and its features"""

    # Load the data, and read the connectivity information:

    triangles = (
        None
        if single_pdb
        else inttensor(np.load(data_dir+'/'+(pdb_id + "_triangles.npy"))).T
    )
    # Normalize the point cloud, as specified by the user:
    points = None if single_pdb else tensor(np.load(data_dir+'/'+(pdb_id + "_xyz.npy")))
    center_location = None if single_pdb else torch.mean(points, axis=0, keepdims=True)

    atom_coords = tensor(np.load(data_dir+'/'+(pdb_id + "_atomxyz.npy")))

    if atom_encoder!=None:
        d=atom_encoder.get('-')
        if d==None:
            d=0
        atom_types=np.load(data_dir+'/'+(pdb_id + "_atomtypes.npy"))
        atom_types_enc=[atom_encoder.get(a, d) for a in atom_types]
        atom_types=inttensor(np.array(atom_types_enc))
        atom_types=F.one_hot(atom_types,num_classes=max(atom_encoder.values())+1).float()
    else:
        atom_types=tensor(np.load(data_dir+'/'+(pdb_id + "_atomtypes.npy")))

    if label_encoder!=None:
        d=label_encoder.get('-')
        if d==None:
            d=0
        atom_res=np.load(data_dir+'/'+(pdb_id + "_resnames.npy"))
        atom_res_enc=[label_encoder.get(a, d) for a in atom_res]
        atom_res=inttensor(np.array(atom_res_enc))
    else:
        atom_res=None


    if center:
        points = points - center_location
        atom_coords = atom_coords - center_location

    # Interface labels
    iface_labels = (
        None
        if single_pdb
        else inttensor(np.load(data_dir+'/'+(pdb_id + "_iface_labels.npy")).reshape((-1, 1)))
    )

    # Features
    chemical_features = (
        None if single_pdb else tensor(np.load(data_dir+'/'+(pdb_id + "_features.npy")))
    )

    # Normals
    normals = (
        None if single_pdb else tensor(np.load(data_dir+'/'+(pdb_id + "_normals.npy")))
    )

    protein_data = Data(
        xyz=points,
        face=triangles,
        chemical_features=chemical_features,
        y=iface_labels,
        normals=normals,
        center_location=center_location,
        num_nodes=None if single_pdb else points.shape[0],
        atom_coords=atom_coords,
        atom_types=atom_types,
        atom_resnames=atom_res,
    )
    return protein_data


class PairData(Data):
    def __init__(
        self,
        xyz_p1=None,
        xyz_p2=None,
        face_p1=None,
        face_p2=None,
        chemical_features_p1=None,
        chemical_features_p2=None,
        y_p1=None,
        y_p2=None,
        normals_p1=None,
        normals_p2=None,
        center_location_p1=None,
        center_location_p2=None,
        atom_coords_p1=None,
        atom_coords_p2=None,
        atom_types_p1=None,
        atom_types_p2=None,
        atom_center1=None,
        atom_center2=None,
        atom_res_p1=None,
        atom_res_p2=None,
        rand_rot1=None,
        rand_rot2=None,
    ):
        super().__init__()
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.face_p1 = face_p1
        self.face_p2 = face_p2

        self.chemical_features_p1 = chemical_features_p1
        self.chemical_features_p2 = chemical_features_p2
        self.y_p1 = y_p1
        self.y_p2 = y_p2
        self.normals_p1 = normals_p1
        self.normals_p2 = normals_p2
        self.center_location_p1 = center_location_p1
        self.center_location_p2 = center_location_p2
        self.atom_coords_p1 = atom_coords_p1
        self.atom_coords_p2 = atom_coords_p2
        self.atom_types_p1 = atom_types_p1
        self.atom_types_p2 = atom_types_p2
        self.atom_center1 = atom_center1
        self.atom_center2 = atom_center2
        self.atom_res_p1 = atom_res_p1
        self.atom_res_p2 = atom_res_p2
        self.rand_rot1 = rand_rot1
        self.rand_rot2 = rand_rot2

    def __inc__(self, key, value, *args, **kwargs):
        if key == "face_p1":
            return self.xyz_p1.size(0)
        if key == "face_p2":
            return self.xyz_p2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if ("index" in key) or ("face" in key):
            return 1
        else:
            return 0


def load_protein_pair(pdb_id, data_dir,single_pdb=False, aa=None, la=None):
    """Loads a protein surface mesh and its features"""
    pspl = pdb_id.split("_")
    p1_id = pspl[0] + "_" + pspl[1]
    p2_id = pspl[0] + "_" + pspl[2]

    p1 = load_protein_npy(p1_id, data_dir, center=False,single_pdb=single_pdb, atom_encoder=aa)
    p2 = load_protein_npy(p2_id, data_dir, center=False,single_pdb=single_pdb, atom_encoder=aa, label_encoder=la)
    # pdist = ((p1['xyz'][:,None,:]-p2['xyz'][None,:,:])**2).sum(-1).sqrt()
    # pdist = pdist<2.0
    # y_p1 = (pdist.sum(1)>0).to(torch.float).reshape(-1,1)
    # y_p2 = (pdist.sum(0)>0).to(torch.float).reshape(-1,1)
    y_p1 = p1.get("y")
    y_p2 = p2.get("y")

    protein_pair_data = PairData(
        xyz_p1=p1.get("xyz"),
        xyz_p2=p2.get("xyz"),
        face_p1=p1.get("face"),
        face_p2=p2.get("face"),
        chemical_features_p1=p1.get("chemical_features"),
        chemical_features_p2=p2.get("chemical_features"),
        y_p1=y_p1,
        y_p2=y_p2,
        normals_p1=p1.get("normals"),
        normals_p2=p2.get("normals"),
        center_location_p1=p1.get("center_location"),
        center_location_p2=p2.get("center_location"),
        atom_coords_p1=p1.get("atom_coords"),
        atom_coords_p2=p2.get("atom_coords"),
        atom_types_p1=p1.get("atom_types"),
        atom_types_p2=p2.get("atom_types"),
        atom_res_p1=p1.get("atom_resnames"),
        atom_res_p2=p2.get("atom_resnames"),
    )
    return protein_pair_data



class NpiDataset(InMemoryDataset):


    def __init__(self, list_file, net, transform=None, binary=False):
        
        with open(list_file) as f_tr:
            self.list = f_tr.read().splitlines()

        self.name=list_file.split('/')[-1].split('.')[0]
       
        self.net=net
        if binary:
            self.la={'-':1 }
            self.name=self.name+'_site'
        else:
            self.la={'DA':1, "DG": 2, "DC":3, "DT":4, '-':0 }
            self.name=self.name+'_npi'

        self.aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "-": 5 }
        
        super().__init__(None, transform, None)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_dir(self) -> str:
        return 'npys/'

    @property
    def processed_file_names(self):
        file_names = [
            self.name+'.pt',
            self.name+'_idx.npy'
        ]

        return file_names

    def process(self):
        
        print('Preprocess', self.name)
        processed_dataset=[]
        processed_idx=[]
        for idx in tqdm(self.list):
            protein_pair = load_protein_pair(idx, 'npys', True, aa=self.aa, la=self.la)

            P1= {}
            P1["atoms"] = protein_pair.atom_coords_p1
            P1["batch_atoms"]=torch.zeros(P1["atoms"].shape[:-1], dtype=torch.int)
            P1["atom_xyz"] = protein_pair.atom_coords_p1
            P1["atomtypes"] = protein_pair.atom_types_p1

            self.net.preprocess_surface(P1)

            P2 = {}
            P2["atoms"] = protein_pair.atom_coords_p2
            P2["batch_atoms"]=torch.zeros(P2["atoms"].shape[:-1], dtype=torch.int)
            P2["atom_xyz"] = protein_pair.atom_coords_p2
            P2["atomtypes"] = protein_pair.atom_types_p2
            P2["atomres"] = protein_pair.atom_res_p2
            
            project_npi_labels(P1, P2, threshold=5.0)
    
            protein_pair.gen_xyz_p1 = P1["xyz"]
            protein_pair.gen_normals_p1 = P1["normals"]
            protein_pair.gen_labels_p1 = P1["labels"]
            protein_pair.gen_batch_p1 = P1["batch"]

            

            if iface_valid_filter(protein_pair):
                processed_dataset.append(protein_pair)
                processed_idx.append(idx)
        processed_dataset, slices=self.collate(processed_dataset)

        torch.save(
            (processed_dataset, slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[1], processed_idx)

