import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
#from lion_pytorch import Lion
import os
import warnings
from data import *
from model import dMaSIF, Lion

import gc
from helper import *


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

    protein_pair_id = protein_pair_id.split(" ")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

    xyz = P["xyz"]

    inputs = P["input_features"]

    embedding = P["embedding_1"] if pdb_idx == 1 else P["embedding_2"]
    emb_id = 1 if pdb_idx == 1 else 2

    if "preds" in P.keys():
        if P["preds"].shape[1]==1:
            predictions = torch.sigmoid(P["preds"])
        else: 
            predictions = F.softmax(P["preds"], dim=1)
    else:
        predictions=torch.zeros((xyz.shape[0],1))

    if predictions.shape[1]==1:
        labels = P["labels"].unsqueeze(dim=1) if P["labels"] is not None else 0.0 * predictions
    else:
        labels = F.one_hot(P["labels"],predictions.shape[1]) if P["labels"] is not None else 0.0 * predictions    
    
    coloring = torch.cat([inputs.to('cpu'), embedding.to('cpu'), predictions.to('cpu'), labels.to('cpu')], axis=1)

    np.save(f'{save_path}/{pdb_id}_predcoords', numpy(xyz))
    np.save(f"{save_path}/{pdb_id}_predfeatures_emb{emb_id}", numpy(coloring))


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
        pos_preds = P1["preds"][P1["labels"] > 0]
        pos_labels = P1["labels"][P1["labels"] > 0]
        neg_preds = P1["preds"][P1["labels"] == 0]
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
                vert=P[key][:,0] if len(P[key].shape)==2 else P[key]
                P[key] = P[key][batch_atoms[vert]]
            else:
                P[key] = P_batch.__getitem__(key)[batch_atoms]
        else:
            if ('face' in key) or ('edge' in key):
                P[key] = P_batch.__getitem__(key)
                vert=P[key][:,0] if len(P[key].shape)==2 else P[key]
                P[key] = P[key][batch[vert]]
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

        if pdb_ids is not None:
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
        info_dict=dict(
                       {
                        'surf_time': outputs["surf_time"],
                        "conv_time": outputs["conv_time"],
                        "memory_usage": outputs["memory_usage"],
                       },
                       # Merge the "R_values" dict into "info", with a prefix:
                       **{"R_values/" + k: v for k, v in outputs["R_values"].items()}
                      )

        P1_batch = outputs["P1"]
        P2_batch = outputs["P2"]

        if P1_batch["labels"] is not None:
            loss, sampled_preds, sampled_labels = compute_loss(args, P1_batch, P2_batch)
            info_dict["Loss"]=loss.detach().item()
        else:
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
            info_dict["ROC-AUC"]=roc_auc
           
        info.append(info_dict)

        if pdb_ids is not None:
            info[-1]['PDB IDs']=batch_ids
            if save_path is not None:
                for i, pdb_id in enumerate(batch_ids):
                    P1 = extract_single(P1_batch, i)
                    P2 = None if args.single_protein else extract_single(P2_batch, i)

                    save_protein_batch_single(
                        pdb_id, P1, save_path, pdb_idx=1
                    )
                    if not args.single_protein:
                        save_protein_batch_single(
                            pdb_id, P2, save_path, pdb_idx=2
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

def ddp_setup(rank, rank_list):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=len(rank_list))
    torch.cuda.set_device(rank_list[rank])

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        args, 
        best_loss = 1e10
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.args = args
        self.args.device=gpu_id
        self.best_loss = best_loss 


    def _run_epoch(self, epoch):

        for dataset_type in ["Train", "Validation", "Test"]:
            if dataset_type == "Train":
                test = False
            else:
                test = True

            suffix = dataset_type
            if dataset_type == "Train":
                dataloader = self.train_loader
            elif dataset_type == "Validation":
                dataloader = self.val_loader
            elif dataset_type == "Test":
                dataloader = self.test_loader
            dataloader.sampler.set_epoch(epoch)

            # Perform one pass through the data:
            info = iterate(
                self.model,
                dataloader,
                self.optimizer,
                self.args,
                test=test,
            )
    
            for key, val in info.items():
                if key in ["Loss", "ROC-AUC"]:
                    print(key ,suffix , epoch, np.nanmean(val))
    
            if dataset_type == "Validation":  # Store validation loss for saving the model
                val_loss = np.nanmean(info["Loss"])
        
                if val_loss < self.best_loss and self.gpu_id==self.args.devices[0]:
                    print("## Validation loss {}, saving model".format(val_loss))
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.module.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_loss": val_loss,
                        },
                        f"models/{self.args.experiment_name}"
                    )
                    self.best_loss = val_loss


    def train(self, starting_epoch: int):
    
        print('# Start training')
        for i in range(starting_epoch, self.args.n_epochs):
            self._run_epoch(i)
            

def load_train_objs(args, net_args):

    net = dMaSIF(net_args)
    optimizer = Lion(net.parameters(), lr=1e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
    starting_epoch = 0
    best_loss = 1e10 


    if args.restart_training != "":
        checkpoint = torch.load("models/" + args.restart_training, map_location=args.devices[0])
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"]+1
        best_loss = checkpoint["best_loss"]

    elif args.transfer_learning != "":
        checkpoint = torch.load("models/" + args.transfer_learning, map_location=args.devices[0])
        for module in checkpoint["model_state_dict"]:
            try:
                net[module].load_state_dict(checkpoint["model_state_dict"][module])
                print('Loaded precomputed module',module)
            except:
                pass 

    print('# Model loaded')
    print('## Model arguments:',net_args)


    transformations = (
        Compose([CenterPairAtoms(as_single=args.search), 
                 RandomRotationPairAtoms(as_single=args.search)])
        if args.random_rotation
        else Compose([])
    )

    pre_transformations=[SurfacePrecompute(net.preprocess_surface, args)]
    if args.search:
        pre_transformations.append(GenerateMatchingLabels(args.threshold))
    else:
        pre_transformations.append(LabelsFromAtoms(single_protein=args.single_protein,
                                                   threshold=args.threshold))
    if args.single_protein:
        pre_transformations.append(RemoveSecondProtein())
    pre_transformations=Compose(pre_transformations)

    print('# Loading datasets')   
    if args.site:
        prefix=f'site_{args.na.lower()}_'
    elif args.npi:
        prefix=f'npi_{args.na.lower()}_'
    elif args.search:
        prefix=f'search_{args.na.lower()}_'
    
    full_dataset = NpiDataset(args.data_dir, args.training_list,
                transform=transformations, pre_transform=pre_transformations, 
                encoders=args.encoders, prefix=prefix, pre_filter=iface_valid_filter)
    test_dataset = NpiDataset(args.data_dir, args.testing_list,
                transform=transformations, pre_transform=pre_transformations,
                encoders=args.encoders, prefix=prefix, pre_filter=iface_valid_filter)

# Train/Validation split:
    train_nsamples = len(full_dataset)
    val_nsamples = int(train_nsamples * args.validation_fraction)
    train_nsamples = train_nsamples - val_nsamples
    train_dataset, val_dataset = random_split(
        full_dataset, [train_nsamples, val_nsamples]
    )
    print('## Train nsamples:',train_nsamples)
    print('## Val nsamples:',val_nsamples)
    print('## Test nsamples:',len(test_dataset))

    return (train_dataset,val_dataset,test_dataset), net, optimizer, starting_epoch, best_loss


def train(rank: int, rank_list: int, args, dataset, net, optimizer, starting_epoch, best_loss):

    warnings.simplefilter("ignore")
    ddp_setup(rank, rank_list)

    batch_vars = ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2"]

    train_loader = DataLoader(
        dataset[0], batch_size=args.batch_size, follow_batch=batch_vars,
        shuffle=False, sampler=DistributedSampler(dataset[0]))
    val_loader = DataLoader(
        dataset[1], batch_size=args.batch_size, follow_batch=batch_vars,
        shuffle=False, sampler=DistributedSampler(dataset[1]))
    test_loader = DataLoader(
        dataset[2], batch_size=1, follow_batch=batch_vars,
        shuffle=False, sampler=DistributedSampler(dataset[2]))


    gc.collect()    
    trainer = Trainer(model=net,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      gpu_id=rank_list[rank],
                      args=args,
                      best_loss=best_loss)
    trainer.train(starting_epoch)

    destroy_process_group()

