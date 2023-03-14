import torch
import numpy as np
from helper import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from sklearn.metrics import roc_auc_score
from pathlib import Path
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from geometry_processing import save_vtk
from helper import numpy, diagonal_ranges
import time
import gc
import warnings 

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.shape[0] == target.shape[0]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        if len(input.shape)==1 or input.shape[1]==1: 
            # binary
            input_soft = torch.sigmoid(input.squeeze())
            input_soft=torch.stack((1. - input_soft, input_soft), dim=1) + self.eps

        else:
        # compute softmax over the classes axis
            input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target.to(torch.int64), num_classes=input_soft.shape[1])

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none') -> torch.Tensor:

    return FocalLoss(alpha, gamma, reduction)(input, target)




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
        )
    else:
        valid2=True

    return valid1 and valid2


def process_single(protein_pair, chain_idx=1):
    """Turn the PyG data object into a dict."""

    P = {}

    if chain_idx == 1:
        with_mesh = "face_p1" in protein_pair.keys
        preprocessed = "gen_xyz_p1" in protein_pair.keys

        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p1 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p1_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p1 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p1 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p1
        P["batch_atoms"] = protein_pair.atom_coords_p1_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p1
        P["atomtypes"] = protein_pair.atom_types_p1
        if "atom_res_p1" in protein_pair.keys:
            P["atomres"] = protein_pair.atom_res_p1

        P["xyz"] = protein_pair.gen_xyz_p1 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p1 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p1 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p1 if preprocessed else None

    elif chain_idx == 2:
        with_mesh = "face_p2" in protein_pair.keys
        preprocessed = "gen_xyz_p2" in protein_pair.keys

                # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p2 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p2_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p2 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p2 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p2
        P["batch_atoms"] = protein_pair.atom_coords_p2_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p2
        P["atomtypes"] = protein_pair.atom_types_p2
        if "atom_res_p2" in protein_pair.keys:
            P["atomres"] = protein_pair.atom_res_p2

        P["xyz"] = protein_pair.gen_xyz_p2 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p2 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p2 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p2 if preprocessed else None

    return P


def save_protein_batch_single(protein_pair_id, P, save_path, pdb_idx):

    protein_pair_id = protein_pair_id.split("_")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

    batch = P["batch"]

    xyz = P["xyz"]

    inputs = P["input_features"]

    embedding = P["embedding_1"] if pdb_idx == 1 else P["embedding_2"]
    emb_id = 1 if pdb_idx == 1 else 2

    if "iface_preds" in P.keys():
        if P["iface_preds"].shape[1]==1:
            predictions = torch.sigmoid(P["iface_preds"])
        else: 
            predictions = F.softmax(P["iface_preds"], dim=1)

    else:
        0.0*embedding[:,0].view(-1, 1)

    if predictions.shape[1]==1:
        labels = P["labels"].unsqueeze(dim=1) if P["labels"] is not None else 0.0 * predictions
    else:
        labels = F.one_hot(P["labels"],predictions.shape[1]) if P["labels"] is not None else 0.0 * predictions


    coloring = torch.cat([inputs, embedding, predictions, labels], axis=1)

    np.save(str(save_path / pdb_id) + "_predcoords", numpy(xyz))
    np.save(str(save_path / pdb_id) + f"_predfeatures_emb{emb_id}", numpy(coloring))


def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1) # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein

    query_labels = labels[nn_i] * nn_dist_i

    P["labels"] = query_labels.detach()

def project_npi_labels(P1, P2, threshold=5.0):

    queries = P1["xyz"]
    batch_queries = P1["batch"]
    source = P2["atom_xyz"]
    batch_source = P2["batch_atoms"]
    labels = P2["atomres"]

    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1).detach()   # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1) < threshold**2
    )  # If chain is not connected because of missing densities MaSIF cut out a part of the protein
    
    query_labels = torch.take(labels,nn_i)
    query_labels=query_labels * nn_dist_i

    P1["labels"] = query_labels.detach()


def process(args, protein_pair, net):
    P1 = process_single(protein_pair, chain_idx=1)
    if not "gen_xyz_p1" in protein_pair.keys:
        if args.random_rotation:
            R1 = tensor(Rotation.random().as_matrix()).to(args.device)
            atom_center1 = P1["atoms"].mean(dim=-2, keepdim=True)
            P1['atoms']=torch.matmul(R1, P1['atoms'].T).T.contiguous()-atom_center1 
            net.preprocess_surface(P1)
            P1['atoms']=torch.matmul(R1.T, (P1['atoms']+atom_center1).T).T.contiguous()
            P1['xyz']=torch.matmul(R1.T, (P1['xyz']+atom_center1).T).T.contiguous()
            P1['normals']=torch.matmul(R1.T, P1['normals'].T).T.contiguous()
        else:
            net.preprocess_surface(P1)
    if P1["mesh_labels"] is not None:
        project_iface_labels(P1)
    elif args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2)
        if P2['atom_xyz'].shape[0]==0:
            P1["labels"] = torch.zeros(P1["xyz"].shape[0]).to(args.device)
        else:
            project_npi_labels(P1, P2, threshold=5.0)
    P2 = None
    if not args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2)
        if not "gen_xyz_p2" in protein_pair.keys:
            if args.random_rotation:
                R2 = tensor(Rotation.random().as_matrix()).to(args.device)
                atom_center2 = P2["atoms"].mean(dim=-2, keepdim=True)
                P2['atoms']=torch.matmul(R2, P2['atoms'].T).T.contiguous()-atom_center2 
                net.preprocess_surface(P2)
                P2['atoms']=torch.matmul(R2.T, (P2['atoms']+atom_center2).T).T.contiguous() 
                P2['xyz']=torch.matmul(R2.T, (P2['xyz']+atom_center2).T).T.contiguous() 
                P2['normals']=torch.matmul(R2.T, P2['normals'].T).T.contiguous()
            else:
                net.preprocess_surface(P2)         
        if P2["mesh_labels"] is not None:
            project_iface_labels(P2)
        elif not args.search:
            project_npi_labels(P2, P1, threshold=5.0)
        else:
            generate_matchinglabels(args, P1, P2)

    return P1, P2


def generate_matchinglabels(args, P1, P2, threshold=4.0):
    if P1.get("atom_center") is not None:
        xyz1_i = torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
    else:
        xyz1_i=P1["xyz"]
    if P2.get("atom_center") is not None:
        xyz2_j = torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
    else:
        xyz2_j=P2["xyz"]
    xyz1_i = LazyTensor(xyz1_i[:, None, :].contiguous())
    xyz2_j = LazyTensor(xyz2_j[None, :, :].contiguous())

    xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1)
    xyz_dists = (threshold**2 - xyz_dists).step()
    p1_iface_labels = (xyz_dists.sum(1) > 1.0).float().view(-1)
    p2_iface_labels = (xyz_dists.sum(0) > 1.0).float().view(-1)

    P1["labels"] = p1_iface_labels
    P2["labels"] = p2_iface_labels


def compute_loss(args, P1, P2, n_points_sample=16, threshold=2):

    if args.search:
        pos_xyz1 = P1["xyz"][P1["labels"] == 1]
        pos_xyz2 = P2["xyz"][P2["labels"] == 1]
        pos_descs1 = P1["embedding_1"][P1["labels"] == 1]
        pos_descs2 = P2["embedding_2"][P2["labels"] == 1]

        pos_xyz_dists = (
            ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1).sqrt()
        )
        pos_desc_dists = torch.matmul(pos_descs1, pos_descs2.T)

        pos_preds = pos_desc_dists[pos_xyz_dists < threshold]
        pos_labels = torch.ones_like(pos_preds)

        n_desc_sample = 100
        sample_desc2 = torch.randperm(len(P2["embedding_2"]))[:n_desc_sample]
        sample_desc2 = P2["embedding_2"][sample_desc2]
        neg_preds = torch.matmul(pos_descs1, sample_desc2.T).view(-1)
        neg_labels = torch.zeros_like(neg_preds)

        # For symmetry
        pos_descs1_2 = P1["embedding_2"][P1["labels"] == 1]
        pos_descs2_2 = P2["embedding_1"][P2["labels"] == 1]

        pos_desc_dists2 = torch.matmul(pos_descs2_2, pos_descs1_2.T)
        pos_preds2 = pos_desc_dists2[pos_xyz_dists.T < threshold]
        pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
        pos_labels = torch.ones_like(pos_preds)

        sample_desc1_2 = torch.randperm(len(P1["embedding_2"]))[:n_desc_sample]
        sample_desc1_2 = P1["embedding_2"][sample_desc1_2]
        neg_preds_2 = torch.matmul(pos_descs2_2, sample_desc1_2.T).view(-1)

        neg_preds = torch.cat([neg_preds, neg_preds_2], dim=0)
        neg_labels = torch.zeros_like(neg_preds)

    else:
        pos_preds = P1["iface_preds"][P1["labels"] > 0]
        pos_labels = P1["labels"][P1["labels"] > 0]
        neg_preds = P1["iface_preds"][P1["labels"] == 0]
        neg_labels = P1["labels"][P1["labels"] == 0]

    n_points_sample = len(pos_labels)
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    if args.npi:
        neg_indices = torch.randperm(len(neg_labels))[:n_points_sample//4]
    else:
        neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]

    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    
    if args.loss=='CELoss':
        loss = F.cross_entropy(preds_concat, labels_concat)
    elif args.loss=='BCELoss':
        loss = F.binary_cross_entropy_with_logits(preds_concat.squeeze(), labels_concat.float())
    elif args.loss=='FocalLoss':
        loss = focal_loss(preds_concat, labels_concat, alpha=0.25, gamma=args.focal_loss_gamma, reduction='mean')


    return loss, preds_concat, labels_concat


def extract_single(P_batch, number):
    P = {}  # First and second proteins
    batch = P_batch["batch"] == number
    batch_atoms = P_batch["batch_atoms"] == number

    with_mesh = P_batch["labels"] is not None
    # Ground truth labels are available on mesh vertices:
    P["labels"] = P_batch["labels"][batch] if with_mesh else None

    P["batch"] = P_batch["batch"][batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][batch]
    P["normals"] = P_batch["normals"][batch]

    # Atom information:
    P["atoms"] = P_batch["atoms"][batch_atoms]
    P["batch_atoms"] = P_batch["batch_atoms"][batch_atoms]

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = P_batch["atom_xyz"][batch_atoms]
    P["atomtypes"] = P_batch["atomtypes"][batch_atoms]


    return P


def iterate(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None,
    pdb_ids=None,
    epoch_number=None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    # Loop over one epoch:
    for it, protein_pair in enumerate(
        tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):
        protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1
        if save_path is not None:
            batch_ids = pdb_ids[
                total_processed_pairs : total_processed_pairs + protein_batch_size
            ]
            total_processed_pairs += protein_batch_size

        protein_pair.to(args.device)

        if not test:
            optimizer.zero_grad()

        # Generate the surface:
        torch.cuda.synchronize()
        surface_time = time.time()
        P1_batch, P2_batch = process(args, protein_pair, net)
        torch.cuda.synchronize()
        surface_time = time.time() - surface_time

        for protein_it in range(protein_batch_size):
            torch.cuda.synchronize()
            iteration_time = time.time()

            P1 = extract_single(P1_batch, protein_it)
            P2 = None if args.single_protein else extract_single(P2_batch, protein_it)

            torch.cuda.synchronize()
            prediction_time = time.time()
            outputs = net(P1, P2)
            torch.cuda.synchronize()
            prediction_time = time.time() - prediction_time

            P1 = outputs["P1"]
            P2 = outputs["P2"]

            #if args.search:
            #    generate_matchinglabels(args, P1, P2)
            
            if P1["labels"] is not None:
                loss, sampled_preds, sampled_labels = compute_loss(args, P1, P2)
            else:
                loss = torch.tensor(0.0)
                sampled_preds = None
                sampled_labels = None

            # Compute the gradient, update the model weights:
            if not test:
                torch.cuda.synchronize()
                back_time = time.time()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                back_time = time.time() - back_time

            if save_path is not None:
                save_protein_batch_single(
                    batch_ids[protein_it], P1, save_path, pdb_idx=1
                )
                if not args.single_protein:
                    save_protein_batch_single(
                        batch_ids[protein_it], P2, save_path, pdb_idx=2
                    )

            if sampled_labels is not None and sampled_labels.shape[0]>0:
                if args.npi:
                    a=np.rint(numpy(sampled_labels))
                    b=numpy(F.softmax(sampled_preds, dim=1))
                    roc_auc = roc_auc_score(
                        a,b, multi_class='ovo', 
                        labels=list(range(args.n_outputs))
                    )
                    
                else:
                    a=np.rint(numpy(sampled_labels.view(-1)))
                    b=numpy(sampled_preds.view(-1))
                    roc_auc = roc_auc_score(a, b)
            else:
                roc_auc = 0.0
           
            R_values = outputs["R_values"]

            info.append(
                dict(
                    {
                        "Loss": loss.detach().item(),
                        "ROC-AUC": roc_auc,
                        "conv_time": outputs["conv_time"],
                        "memory_usage": outputs["memory_usage"],
                    },
                    # Merge the "R_values" dict into "info", with a prefix:
                    **{"R_values/" + k: v for k, v in R_values.items()}                )
            )
            torch.cuda.synchronize()
            iteration_time = time.time() - iteration_time


    # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)

    info = newdict

    gc.collect()
    torch.cuda.empty_cache()
    # Final post-processing:
    return info


class SurfacePrecompute(object):
    r"""Precomputation of surface"""

    def __init__(self, net, args):
        self.args=args
        self.net=net

    def __call__(self, protein_pair):
        

        if 'xyz_p1' in protein_pair.keys:
            protein_pair.xyz_p1_batch=torch.zeros(protein_pair.xyz_p1.shape[:-1], dtype=torch.int)
        protein_pair.atom_coords_p1_batch=torch.zeros(protein_pair.atom_coords_p1.shape[:-1], dtype=torch.int)
        if 'xyz_p2' in protein_pair.keys:
            protein_pair.xyz_p2_batch=torch.zeros(protein_pair.xyz_p2.shape[:-1], dtype=torch.int)
        protein_pair.atom_coords_p2_batch=torch.zeros(protein_pair.atom_coords_p2.shape[:-1], dtype=torch.int)

        protein_pair.to(self.args.device)

        P1, P2 = process(self.args, protein_pair, self.net)
        protein_pair.gen_xyz_p1 = P1["xyz"]
        protein_pair.gen_normals_p1 = P1["normals"]
        protein_pair.gen_batch_p1 = P1["batch"]
        protein_pair.gen_labels_p1 = P1["labels"]

        if not self.args.single_protein:
            protein_pair.gen_xyz_p2 = P2["xyz"]
            protein_pair.gen_normals_p2 = P2["normals"]
            protein_pair.gen_batch_p2 = P2["batch"]
            protein_pair.gen_labels_p2 = P2["labels"]
        else: 
            protein_pair.xyz_p2=None
            protein_pair.face_p2=None
            protein_pair.chemical_features_p2=None
            protein_pair.y_p2=None
            protein_pair.normals_p2=None
            protein_pair.center_location_p2=None
            protein_pair.atom_coords_p2=None
            protein_pair.atom_types_p2=None
            protein_pair.atom_res_p2=None
            protein_pair.gen_xyz_p2 = None
            protein_pair.gen_normals_p2 = None
            protein_pair.gen_batch_p2 = None
            protein_pair.gen_labels_p2 = None
        protein_pair.to("cpu")
        return protein_pair


    def __repr__(self):
        return "{}()".format(self.__class__.__name__)