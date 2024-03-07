import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from pykeops.torch import LazyTensor

from geometry_processing import (
    curvatures,
    mesh_normals_areas,
    tangent_vectors,
    atoms_to_points_normals,
)
from helper import *
from geometry_processing import dMaSIFConv_seg

# create Adam optimizer class from https://github.com/lucidrains/lion-pytorch


from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None

# class

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                p.data.mul_(1 - lr * wd)

                # weight update

                update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
                p.add_(update, alpha = -lr)

                # decay the momentum running average coefficient

                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

        return loss

def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, x_batch, y_batch, y_atomtype, k=16):

    idx, dists = knn_atoms(x, y, x_batch, y_batch, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature

def get_features_v(x, y, x_batch, y_batch, y_atomtype, k=16, gamma=1):

    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1).view(-1)  # (N, K)
    
    x_ik = y[idx,:].view(N, k, D)

    vecs=(x[:, None, :]-x_ik) # (N, K, D)   
    
    dists = (vecs ** 2).sum(-1)

    dists = torch.pow(dists+1e-8,-(1+gamma)/2) # (N, K)
    
    _, num_dims = y_atomtype.size()

    #normalize coords by distance
    norm_vec=vecs*dists[:, :, None] #(N, k, D)
    
    feature = y_atomtype[idx, :].view(N, k, num_dims)

    feature=norm_vec[:,:,:,None] * feature[:,:,None,:] # (N, k, D, num_dims )


    return torch.transpose(feature, 1, 3) # (N, num_dims, D, k )

class AtomNet_V(nn.Module):
    def __init__(self, args):
        super(AtomNet_V, self).__init__()
        self.atom_dims=args['atom_dims']
        self.k = args['knn']
        self.transform_types = nn.Sequential(
            nn.Linear(args['atom_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dropout=nn.Dropout2d(args['dropout'])

        self.att=nn.Sequential(
            nn.Linear(self.k, 1, bias=False))
        
        self.embedding = nn.Sequential(
            nn.Linear(args['chem_dims'],args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.BatchNorm1d(args['chem_dims']),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.BatchNorm1d(args['chem_dims']),
            nn.Linear(args['chem_dims'], args['chem_dims']),
        )



    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):

        atomtypes=atomtypes[:,:self.atom_dims]
        atomtypes = self.transform_types(atomtypes)
        fx = get_features_v(xyz, atom_xyz, batch, atom_batch, atomtypes, k=self.k)
        fx = self.att(self.dropout(fx)).squeeze(-1)
        fx= torch.sqrt(torch.square(fx).sum(dim=-1, keepdim=False))
        fx = self.embedding(fx)
        return fx

class AtomNet_V_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_V_MP, self).__init__()
        self.atom_dims=args['atom_dims']
        self.k = args['knn']
        self.transform_types = nn.Sequential(
            nn.Linear(args['atom_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims'])
        )

        self.transform_types_mp = nn.Sequential(
            nn.Linear(args['atom_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims'])
        )
        self.att=nn.Sequential(
            nn.Linear(self.k, 1, bias=False))

        self.att_mp=nn.Sequential(
            nn.Linear(self.k, 1, bias=False))

        self.bil=nn.Bilinear(args['chem_dims'],args['chem_dims'],args['chem_dims'], bias=False)

        self.dropout=nn.Dropout(args['dropout'])
        self.dropout_mp=nn.Dropout(args['dropout'])
        
        self.embedding_mp = nn.Sequential(
            nn.Linear(args['chem_dims'],args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
        )
        self.embedding = nn.Sequential(
            nn.Linear(args['chem_dims'],args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
        )

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):

        atomtypes=atomtypes[:,:self.atom_dims]

        fx = self.transform_types_mp(atomtypes)
        fx = get_features_v(atom_xyz, atom_xyz, atom_batch, atom_batch, fx, k=self.k+1)
        fx=fx[:,:,:,1:] # Remove self
        fx = self.att_mp(self.dropout_mp(fx)).squeeze(-1)
        fx= torch.sqrt(torch.square(fx).sum(dim=-1, keepdim=False))
        fx = self.embedding_mp(fx)
       
        atomtypes=self.transform_types(atomtypes)

        atomtypes=atomtypes-self.bil(atomtypes,fx)

        fx = get_features_v(xyz, atom_xyz, batch, atom_batch, atomtypes, k=self.k)
        fx = self.att(self.dropout(fx)).squeeze(-1)
        fx= torch.sqrt(torch.square(fx).sum(dim=-1, keepdim=False))
        fx = self.embedding(fx)

        return fx


class Atom_embedding(nn.Module):
    def __init__(self, args):
        super(Atom_embedding, self).__init__()
        self.D = args['chem_dims']
        self.k = 16
        self.dropout=nn.Dropout(args['dropout'])
        self.conv1 = nn.Linear(self.D + 1, self.D)
        self.conv2 = nn.Linear(self.D, self.D)
        self.conv3 = nn.Linear(2 * self.D, self.D)
        self.bn1 = nn.BatchNorm1d(self.D)
        self.bn2 = nn.BatchNorm1d(self.D)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        fx = get_atom_features(x, y, x_batch, y_batch, y_atomtypes, k=self.k)
        fx = self.dropout(fx)
        fx = self.conv1(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn1(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx1 = fx.sum(dim=1, keepdim=False)

        fx = self.conv2(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn2(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx2 = fx.sum(dim=1, keepdim=False)
        fx = torch.cat((fx1, fx2), dim=-1)
        fx = self.conv3(fx)

        return fx


class AtomNet(nn.Module):
    def __init__(self, args):
        super(AtomNet, self).__init__()
        self.atom_dims=args['atom_dims']

        self.transform_types = nn.Sequential(
            nn.Linear(args['atom_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['chem_dims'], args['chem_dims']),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.embed = Atom_embedding(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes=atomtypes[:,:self.atom_dims]
        atomtypes = self.transform_types(atomtypes)
        return self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

class Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_MP, self).__init__()
        self.D = args['chem_dims']
        self.k = 16
        self.n_layers = 3
        self.dropout=nn.Dropout(args['dropout'])
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.dropout(self.mlp[i](features))  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb

class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.D = args['chem_dims']
        self.k = 17
        self.n_layers = 3
        self.dropout=nn.Dropout(args['dropout'])

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.dropout(self.mlp[i](features))  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out

class AtomNet_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP, self).__init__()
        self.atom_dims=args['atom_dims']

        self.transform_types = nn.Sequential(
            nn.Linear(args['atom_dims'], args['atom_dims']),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args['atom_dims'], args['chem_dims']),
        )

        self.embed = Atom_embedding_MP(args)
        self.atom_atom = Atom_Atom_embedding_MP(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes=atomtypes[:,:self.atom_dims]
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)
        return atomtypes


def combine_pair(P1, P2):
    P1P2 = {}
    for key in P1:
        v1 = P1[key]
        v2 = P2[key]
        if v1 is None:
            continue

        if 'batch' in key:
            v1v2 = torch.cat([v1*2, v2*2 + 1], dim=0)
        elif ("face" in key) or ('edge' in key):
            v1v2 = torch.cat([v1, v2 + P1['xyz'].shape[0]], dim=0)
        else:
            v1v2 = torch.cat([v1, v2], dim=0)
        P1P2[key] = v1v2

    return P1P2


def split_pair(P1P2):
    p1_indices = (P1P2["batch_xyz"] % 2) == 0
    p2_indices = (P1P2["batch_xyz"] % 2) == 1

    p1_atom_indices = (P1P2["batch_atom_xyz"] % 2) == 0
    p2_atom_indices = (P1P2["batch_atom_xyz"] % 2) == 1

    P1 = {}
    P2 = {}
    for key in P1P2:
        v1v2 = P1P2[key]

        if (key == "rand_rot") or (key == "atom_center"):
            n = v1v2.shape[0] // 2
            P1[key] = v1v2[:n].view(-1, 3)
            P2[key] = v1v2[n:].view(-1, 3)
        elif "atom" in key:
            P1[key] = v1v2[p1_atom_indices]
            P2[key] = v1v2[p2_atom_indices]
        elif ("face" in key) or ('edge' in key):
            P1[key] = v1v2[v1v2<sum(p1_indices)]
            P2[key] = v1v2[v1v2>=sum(p1_indices)]-sum(p1_indices)
        else:
            P1[key] = v1v2[p1_indices]
            P2[key] = v1v2[p2_indices]
            if 'batch' in key:
                P1[key] = P1[key] // 2
                P2[key] = P2[key] // 2

    return P1, P2


class dMaSIF(nn.Module):
    def __init__(self, args):
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = args['curvature_scales']
        self.args = args

        I = len(args['curvature_scales'])*2+args['chem_dims']
        O = args['orientation_units']
        E = args['emb_dims']
        H = args['post_units']
        C = args['n_outputs']

        # Computes chemical features
        if args['feature_generation']=='AtomNet':
            self.atomnet = AtomNet(args)
        elif args['feature_generation']=='AtomNet_MP':
            self.atomnet = AtomNet_MP(args)
        elif args['feature_generation']=='AtomNet_V':
            self.atomnet = AtomNet_V(args)
        elif args['feature_generation']=='AtomNet_V_MP':
            self.atomnet = AtomNet_V_MP(args)

        self.dropout = nn.Dropout(args['dropout'])

            # Post-processing, without batch norm:
        self.orientation_scores = nn.Sequential(
                nn.Linear(I, O),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(O, 1),
            )

            # Segmentation network:
        self.conv = dMaSIFConv_seg(
                args,
                in_channels=I,
                out_channels=E,
                n_layers=args['n_layers'],
                radius=args['radius'],
            )

            # Asymmetric embedding
        if args['split']:
            self.orientation_scores2 = nn.Sequential(
                    nn.Linear(I, O),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(O, 1),
                )

            self.conv2 = dMaSIFConv_seg(
                    args,
                    in_channels=I,
                    out_channels=E,
                    n_layers=args['n_layers'],
                    radius=args['radius'],
                )

        if C>0:
            # Post-processing, without batch norm:
            self.net_out = nn.Sequential(
                nn.Linear(E, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, C),
            )


    def features(self, P, i=1):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

        # Estimate the curvatures using the triangles or the estimated normals:
        P_curvatures = curvatures(
            P["xyz"],
            triangles=None,
            normals= P["normals"],
            scales=self.curvature_scales,
            batch=P["batch_xyz"],
        )

        # Compute chemical features on-the-fly:
        chemfeats = self.atomnet(
            P["xyz"], P["atom_xyz"], P["atom_types"], P["batch_xyz"], P["batch_atom_xyz"]
        )

        # Concatenate our features:
        return torch.cat([P_curvatures, chemfeats], dim=1).contiguous()

    def embed(self, P):
        """Embeds all points of a protein in a high-dimensional vector space."""

        features = self.dropout(self.features(P))
        P["input_features"] = features
        
        conv_time = time.time()

        # Ours:
        self.conv.load_mesh(
                P["xyz"],
                triangles=None,
                normals=P["normals"],
                weights=self.orientation_scores(features),
                batch=P["batch_xyz"],
            )
        P["embedding_1"] = self.conv(features)
        if self.args['split']:
            self.conv2.load_mesh(
                    P["xyz"],
                    triangles=None,
                    normals=P["normals"],
                    weights=self.orientation_scores2(features),
                    batch=P["batch_xyz"],
                )
            P["embedding_2"] = self.conv2(features)

        conv_time = time.time()-conv_time
        memory_usage = torch.cuda.max_memory_allocated()

        return conv_time, memory_usage

    def preprocess_surface(self, P):
        surf_time = time.time()

        if 'batch_atom_xyz' not in P.keys():
            P["batch_atom_xyz"]=torch.zeros(P["atom_xyz"].shape[0],device=P["atom_xyz"].device, dtype=int)
            
        P["xyz"], P["normals"], P["batch_xyz"] = atoms_to_points_normals(
            P["atom_xyz"],
            P["batch_atom_xyz"],
            atom_rad=P["atom_rad"],
            resolution=self.args['resolution'],
            sup_sampling=self.args['sup_sampling'],
            distance=self.args['distance']
        )

        surf_time = time.time()-surf_time

        return surf_time

    def forward(self, P1, P2=None):
        # Compute embeddings of the point clouds:
        surf_time=0
        if ("xyz" not in P1):
            surf_time = self.preprocess_surface(P1)

        if P2 is not None:
            if ("xyz" not in P2):
                surf_time += self.preprocess_surface(P2)
            P1P2 = combine_pair(P1, P2)
        else:
            P1P2 = P1

        conv_time, memory_usage = self.embed(P1P2)

        # Monitor the approximate rank of our representations:
        R_values = {}
        R_values["input"] = soft_dimension(P1P2["input_features"])
        R_values["conv"] = soft_dimension(P1P2["embedding_1"])

        if self.args['n_outputs']>0:
            P1P2["preds"] = self.net_out(P1P2["embedding_1"])

        if P2 is not None:
            P1, P2 = split_pair(P1P2)
        else:
            P1 = P1P2

        return {
            "P1": P1,
            "P2": P2,
            "R_values": R_values,
            "surf_time": surf_time,        
            "conv_time": conv_time,
            "memory_usage": memory_usage,
        }
