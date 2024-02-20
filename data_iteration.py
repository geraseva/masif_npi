import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import gc
from config import *


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
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
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'none') -> torch.Tensor:

    return FocalLoss(alpha, gamma, reduction)(input, target)


def save_protein_batch_single(protein_pair_id, P, save_path, pdb_idx):

    protein_pair_id = protein_pair_id.split("_")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

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
        predictions=torch.zeros((xyz.shape[0],1))

    if predictions.shape[1]==1:
        labels = P["labels"].unsqueeze(dim=1) if P["labels"] is not None else 0.0 * predictions
    else:
        labels = F.one_hot(P["labels"],predictions.shape[1]) if P["labels"] is not None else 0.0 * predictions    
    
    coloring = torch.cat([inputs.to('cpu'), embedding.to('cpu'), predictions.to('cpu'), labels.to('cpu')], axis=1)

    np.save(str(save_path / pdb_id) + "_predcoords", numpy(xyz))
    np.save(str(save_path / pdb_id) + f"_predfeatures_emb{emb_id}", numpy(coloring))


def compute_loss(args, P1, P2):

    if args.search:

        pos_descs1 = P1["embedding_1"][P1["edge_labels"],:]
        pos_descs2 = P2["embedding_2"][P2["edge_labels"],:]
        pos_preds = torch.sum(pos_descs1*pos_descs2, axis=-1)

        pos_descs1_2 = P1["embedding_2"][P1["edge_labels"],:]
        pos_descs2_2 = P2["embedding_1"][P2["edge_labels"],:]
        pos_preds2 = torch.sum(pos_descs1_2*pos_descs2_2, axis=-1)

        pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)

        n_desc_sample = 100
        
        sample_desc1=P1["embedding_1"][P1["labels"] == 1]
        sample_desc2 = torch.randperm(len(P2["embedding_2"]))[:n_desc_sample]
        sample_desc2 = P2["embedding_2"][sample_desc2]
        neg_preds = torch.matmul(sample_desc1, sample_desc2.T).view(-1)

        sample_desc2_1=P1["embedding_2"][P1["labels"] == 1]
        sample_desc1_2 = torch.randperm(len(P1["embedding_2"]))[:n_desc_sample]
        sample_desc1_2 = P1["embedding_2"][sample_desc1_2]
        neg_preds_2 = torch.matmul(sample_desc2_1, sample_desc1_2.T).view(-1)

        neg_preds = torch.cat([neg_preds, neg_preds_2], dim=0)

        pos_labels = torch.ones_like(pos_preds)
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
        loss = F.cross_entropy(preds_concat, labels_concat, reduction='mean')
    elif args.loss=='BCELoss':
        loss = F.binary_cross_entropy_with_logits(preds_concat.squeeze(), labels_concat.float(),
                                                  reduction='mean')
    elif args.loss=='FocalLoss':
        loss = focal_loss(preds_concat, labels_concat, reduction='mean')


    return loss, preds_concat, labels_concat

def extract_single(P_batch, number):
    P = {}  # First and second proteins
    batch = P_batch["xyz_batch"] == number
    batch_atoms = P_batch["atom_xyz_batch"] == number

    for key in P_batch.keys():
        if 'atom' in key:
            if ('face' in key) or ('edge' in key):
                P[key] = P_batch.__getitem__(key)
                P[key] = P[key][batch_atoms[P[key][:,0]],:]
            else:
                P[key] = P_batch.__getitem__(key)[batch_atoms]
        else:
            if ('face' in key) or ('edge' in key):
                P[key] = P_batch.__getitem__(key)
                P[key] = P[key][batch[P[key][:,0]],:]
            else:
                P[key] = P_batch.__getitem__(key)[batch]
    return P


def iterate(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None,
    pdb_ids=None,
):

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
    for protein_pair in tqdm(dataset):  

        if save_path is not None:
            batch_ids = pdb_ids[
                total_processed_pairs : total_processed_pairs + args.batch_size
            ]
            total_processed_pairs += args.batch_size
        protein_pair.to(args.device)
        
        if not test:
            optimizer.zero_grad()

        P1_batch = protein_pair.to_dict(chain_idx=1)
        P2_batch = None if args.single_protein else  protein_pair.to_dict(chain_idx=2)
 
        outputs = net(P1_batch, P2_batch)

        P1_batch = outputs["P1"]
        P2_batch = outputs["P2"]

        if P1_batch["labels"] is not None:
            loss, sampled_preds, sampled_labels = compute_loss(args, P1_batch, P2_batch)
        else:
            loss = torch.tensor(0.0)
            sampled_preds = None
            sampled_labels = None

        # Compute the gradient, update the model weights:
        if not test:
            loss.backward()
            optimizer.step()   

        if sampled_labels is not None and sampled_labels.shape[0]>0:
            if len(sampled_preds.shape)>1 and sampled_preds.shape[1]>1:
                a=np.rint(numpy(sampled_labels))
                b=numpy(F.softmax(sampled_preds, dim=1))
                roc_auc = roc_auc_score(
                    a,b, multi_class='ovo', 
                    labels=list(range(sampled_preds.shape[1]))
                )
            else:
                a=np.rint(numpy(sampled_labels.view(-1)))
                b=numpy(sampled_preds.view(-1))
                roc_auc = roc_auc_score(a, b)
        else:
            roc_auc = 0.0
           
        info.append(
            dict(
                {
                    "Loss": loss.detach().item(),
                    "ROC-AUC": roc_auc,
                    'surf_time': outputs["surf_time"],
                    "conv_time": outputs["conv_time"],
                    "memory_usage": outputs["memory_usage"],
                },
                # Merge the "R_values" dict into "info", with a prefix:
                **{"R_values/" + k: v for k, v in outputs["R_values"].items()}
                )
        )

        if save_path is not None:
            for protein_it in range(args.batch_size):
                P1 = extract_single(P1_batch, protein_it)
                P2 = None if args.single_protein else extract_single(P2_batch, protein_it)

                save_protein_batch_single(
                    batch_ids[protein_it], P1, save_path, pdb_idx=1
                )
                if not args.single_protein:
                    save_protein_batch_single(
                        batch_ids[protein_it], P2, save_path, pdb_idx=2
                    )


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



