import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Dataset, Data
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from pathlib import Path

from helper import *


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



def load_protein_npy(pdb_id, data_dir, encoders, use_surfaces=False):

    list_to_onehot=['atom_types']

    protein_data={}
    protein_data['atom_coords']=tensor(np.load(data_dir+'/'+(pdb_id + "_atomxyz.npy")))

    atom_types=np.load(data_dir+'/'+(pdb_id + "_atomtypes.npy"))
    atom_types=[a[0] for a in atom_types]

    for aa in encoders['atom_encoders']:
        o=max(aa['encoder'].values())+1 if aa['name'] in list_to_onehot else 0
        protein_data[aa['name']] = encode_labels(atom_types,aa['encoder'],o)
    
    if encoders.get('residue_encoders') != None:
        try:
            atom_res=np.load(data_dir+'/'+(pdb_id + "_resnames.npy"))
        except FileNotFoundError:
            atom_res=np.zeros(len(atom_types))
        for la in encoders['residue_encoders']:
            protein_data[la['name']] = encode_labels(atom_res,la['encoder'])    

    mask=torch.ones(protein_data['atom_coords'].shape[0], dtype=bool)
    for key in protein_data:
        if 'mask' in key:
            mask=mask&protein_data.pop[key]
    for key in protein_data:
        protein_data[key]=protein_data[key][mask]


    protein_data['face'] = (
        inttensor(np.load(data_dir+'/'+(pdb_id + "_triangles.npy"))).T
        if use_surfaces
        else None
    )
    protein_data['xyz'] = tensor(np.load(data_dir+'/'+(pdb_id + "_xyz.npy"))) if use_surfaces else None

    # Interface labels
    protein_data['iface_labels'] = (
        inttensor(np.load(data_dir+'/'+(pdb_id + "_iface_labels.npy")).reshape((-1, 1)))
        if use_surfaces
        else None
    )

    # Features
    protein_data['chemical_features'] = (
        tensor(np.load(data_dir+'/'+(pdb_id + "_features.npy"))) if use_surfaces else None
    )

    # Normals
    protein_data['normals'] = (
        tensor(np.load(data_dir+'/'+(pdb_id + "_normals.npy"))) if use_surfaces else None
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
        atom_center1=None,
        atom_center2=None,
        atom_res_p1=None,
        atom_res_p2=None,
        atom_rad_p1=None,
        atom_rad_p2=None,
        rand_rot1=None,
        rand_rot2=None,
        edge_labels_p1=None,
        edge_labels_p2=None  
          ):
        super().__init__()
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.face_p1 = face_p1
        self.face_p2 = face_p2
        self.chemical_features_p1 = chemical_features_p1
        self.chemical_features_p2 = chemical_features_p2
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
        self.atom_center1 = atom_center1
        self.atom_center2 = atom_center2
        self.atom_res_p1 = atom_res_p1
        self.atom_res_p2 = atom_res_p2
        self.atom_rad_p1 = atom_rad_p1
        self.atom_rad_p2 = atom_rad_p2
        self.rand_rot1 = rand_rot1
        self.rand_rot2 = rand_rot2

    def __inc__(self, key, value, *args, **kwargs):
        if ('face' in key) or ('edge' in key):
            if key[-3:]== '_p1':
                return self.xyz_p1.size(0)
            else:
                return self.xyz_p2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)



def load_protein_pair(pdb_id, data_dir,use_surfaces=False, encoders=None):
    """Loads a protein surface mesh and its features"""
    pspl = pdb_id.split("_")

    p1_id = pspl[0] + "_" + pspl[1]
    p1 = load_protein_npy(p1_id, data_dir,use_surfaces=use_surfaces, encoders=encoders)
    
    try:
        p2_id = pspl[0] + "_" + pspl[2]
    except IndexError:
        p2={}
    else:
        p2 = load_protein_npy(p2_id, data_dir,use_surfaces=use_surfaces, encoders=encoders)


    protein_pair_data = PairData(
        xyz_p1=p1.get("xyz"),
        xyz_p2=p2.get("xyz"),
        face_p1=p1.get("face"),
        face_p2=p2.get("face"),
        chemical_features_p1=p1.get("chemical_features"),
        chemical_features_p2=p2.get("chemical_features"),
        labels_p1=p1.get("iface_labels"),
        labels_p2=p2.get("iface_labels"),
        normals_p1=p1.get("normals"),
        normals_p2=p2.get("normals"),
        atom_xyz_p1=p1.get("atom_coords"),
        atom_xyz_p2=p2.get("atom_coords"),
        atom_types_p1=p1.get("atom_types"),
        atom_types_p2=p2.get("atom_types"),
        atom_res_p1=p1.get("atom_resnames"),
        atom_res_p2=p2.get("atom_resnames"),
        atom_rad_p1=p1.get("atom_rad"),
        atom_rad_p2=p2.get("atom_rad")
    )
    return protein_pair_data


class NpiDataset(InMemoryDataset):


    def __init__(self, root, list_file, encoders, prefix='', use_surfaces=True,
        transform=None, pre_transform=None, pre_filter=None):
        
        with open(list_file) as f_tr:
            self.list = f_tr.read().splitlines()

        self.name=prefix+list_file.split('/')[-1].split('.')[0]
        self.encoders=encoders
        self.use_surfaces=use_surfaces
        
        super(NpiDataset, self).__init__(root, transform, pre_transform,pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0], map_location=default_device)
    

    @property
    def processed_file_names(self):
        file_names = [
            self.name+'.pt',
            self.name+'_idx.npy'
        ]

        return file_names

    def process(self):
        
        print('# Loading npy files', self.name)

        protein_dir = str(Path(self.root))+'/raw/01-benchmark_surfaces_npy/'
        processed_dataset=[]
        processed_idx=[]
        for idx in tqdm(self.list):
            try:
                protein_pair = load_protein_pair(idx, protein_dir, use_surfaces=self.use_surfaces, 
                    encoders=self.encoders)
            except FileNotFoundError:
                print(f'##! Skipping non-existing files for {idx}' )
                continue
            processed_dataset.append(protein_pair)
            processed_idx.append(idx)
        if self.pre_transform is not None:

            print('Preprocessing files', self.name)
            processed_dataset = [
                self.pre_transform(data) for data in tqdm(processed_dataset)
            ]

        if self.pre_filter is not None:
            processed_dataset = [
                data.to('cpu') for data in processed_dataset if self.pre_filter(data)
            ]
        
        processed_dataset, slices=self.collate(processed_dataset)

        torch.save(
            (processed_dataset, slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[1], processed_idx)



class SurfacePrecompute(object):
    r"""Precomputation of surface"""

    def __init__(self, surf_gen, args):
        self.args=args
        self.preprocess_surface=surf_gen

    def __call__(self, protein_pair):

        P1 = {}
        P1["atom_xyz"] = protein_pair.atom_xyz_p1
        P1["atom_rad"] = protein_pair.atom_rad_p1
        P1['atom_xyz_batch']=torch.zeros(P1['atom_xyz'].shape[0], dtype=torch.int).to(self.args.device)
        if self.args.random_rotation:
            R1 = tensor(Rotation.random().as_matrix())
            atom_center1 = P1["atom_xyz"].mean(dim=-2, keepdim=True)
            P1['atom_xyz']=torch.matmul(R1, P1['atom_xyz'].T).T.contiguous()-atom_center1 
            self.preprocess_surface(P1)
            P1['xyz']=torch.matmul(R1.T, (P1['xyz']+atom_center1).T).T.contiguous()
            P1['normals']=torch.matmul(R1.T, P1['normals'].T).T.contiguous()
        else:
            self.preprocess_surface(P1)
        protein_pair.gen_xyz_p1 = P1["xyz"]
        protein_pair.gen_normals_p1 = P1["normals"]
        protein_pair.gen_face_p1 = P1.get("face")

        if not self.args.single_protein:
            P2 = {}
            P2["atom_xyz"] = protein_pair.atom_xyz_p2
            P2["atom_rad"] = protein_pair.atom_rad_p2
            P2['atom_xyz_batch']=torch.zeros(P2['atom_xyz'].shape[0], dtype=torch.int).to(self.args.device)
            if self.args.random_rotation:
                R2 = tensor(Rotation.random().as_matrix())
                atom_center2 = P2["atom_xyz"].mean(dim=-2, keepdim=True)
                P2['atom_xyz']=torch.matmul(R2, P2['atom_xyz'].T).T.contiguous()-atom_center2 
                self.preprocess_surface(P2)
                P2['xyz']=torch.matmul(R2.T, (P2['xyz']+atom_center2).T).T.contiguous() 
                P2['normals']=torch.matmul(R2.T, P2['normals'].T).T.contiguous()
            else:
                self.preprocess_surface(P2)      
            protein_pair.gen_xyz_p2 = P2["xyz"]
            protein_pair.gen_normals_p2 = P2["normals"]
            protein_pair.gen_face_p2 = P2.get("face")
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

class TransferSurface(object):
    r"""Relabel surface points by outsource mesh"""

    def __init__(self, single_protein=True, threshold=2.0):
        self.threshold=threshold
        self.single=single_protein
        self.relabel_list=['labels','chemical_features']
        self.surface_list=['xyz','normals','face']

    def __call__(self, protein_pair):

        if 'xyz_p1' in protein_pair.keys:
            for key in self.relabel_list:
                if f'{key}_p1' in protein_pair.keys:
                    query_labels=get_threshold_labels(queries = protein_pair.gen_xyz_p1,
                                                      batch_queries = None,
                                                      source = protein_pair.xyz_p1,
                                                      batch_source = None,
                                                      labels = protein_pair.__getitem__(f'{key}_p1'),
                                                      threshold=self.threshold)
                    protein_pair.__setitem__(f'{key}_p1',query_labels.detach())

        if not self.single:
            if 'xyz_p2' in protein_pair.keys:
                for key in self.relabel_list:
                    if f'{key}_p2' in protein_pair.keys:
                        query_labels=get_threshold_labels(queries = protein_pair.gen_xyz_p2,
                                                          batch_queries = None,
                                                          source = protein_pair.xyz_p2,
                                                          batch_source = None,
                                                          labels = protein_pair.__getitem__(f'{key}_p2'),
                                                          threshold=self.threshold)
                        protein_pair.__setitem__(f'{key}_p2',query_labels.detach())

        #remove old surfaces and set new
        for key in protein_pair.keys:
            if key[:-3] in self.surface_list:
                protein_pair.__delitem__(key)
        for key in protein_pair.keys:        
            if key[:4]=='gen_':
                protein_pair.__setitem__(key[4:],protein_pair.__getitem__(key))
                protein_pair.__delitem__(key)
        return protein_pair

class LabelsFromAtoms(object):
    r"""Label surface points by complementary atom labels"""

    def __init__(self, threshold=5, single_protein=True):
        self.threshold=threshold
        self.single=single_protein

    def __call__(self, protein_pair):

        if protein_pair.atom_xyz_p2.shape[0]==0:
            query_labels=torch.zeros(queries.shape[0])
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
            if source.shape[0]==0:
                query_labels=torch.zeros(queries.shape[0])
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
            if key[-3:]=='_p2':
                protein_pair.__delitem__(key)

        return protein_pair


class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __init__(self, as_single=False):

        self.as_single=as_single

    def __call__(self, data):

        R1 = tensor(Rotation.random().as_matrix())
        if self.as_single:
            R2=R1
        else:
            R2 = tensor(Rotation.random().as_matrix())

        data.rand_rot1 = R1

        for key in data.keys: 
            if (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p1':
                data.__setitem__(key,torch.matmul(R1, data.__getitem__(key).T).T)
            elif (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p2':
                data.__setitem__(key,torch.matmul(R1, data.__getitem__(key).T).T)
                data.rand_rot2 = R2
          
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
            if (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p1':
                data.__setitem__(key, data.__getitem__(key) - atom_center1)
            elif (('xyz' in key) or ('normals' in key)) and key[-3:]=='_p2':
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
    
    labels2 = protein_pair.get('labels_p2')
    if labels2 != None:
        labels2 = labels2.reshape(-1)>0
        valid2 = (
            (torch.sum(labels2) < 0.75 * len(labels2))
            and (torch.sum(labels2) > 30)
        )
    else:
        valid2=True

    return valid1 and valid2