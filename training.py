# Standard imports:
import numpy as np
import torch
import json
import os, sys
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from lion_pytorch import Lion
from pathlib import Path

# Custom data loader and model:
from data import NpiDataset, PairData, CenterPairAtoms, ProteinPairsSurfaces
from data import RandomRotationPairAtoms
from model import dMaSIF
from data_iteration import iterate, iface_valid_filter, SurfacePrecompute
from helper import *
from Arguments import parse_train
import gc
import warnings
warnings.filterwarnings("ignore")

# Parse the arguments, prepare the TensorBoard writer:
args, net_args = parse_train()

print('Start training')
print('Arguments:',args)
print('Model arguments:',net_args)

# Ensure reproducibility:
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.device!='cpu':
    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)


batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
model_path = "models/" + args.experiment_name
transformations = (
    Compose([CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else None
)
    
net = dMaSIF(net_args)
net = net.to(args.device)

if args.na=='protein':
    full_dataset = ProteinPairsSurfaces(
        "surface_data", ppi=args.search, train=True, transform=transformations, 
        pre_transform=SurfacePrecompute(net, args), pre_filter=iface_valid_filter,
        aa=args.aa)
    test_dataset = ProteinPairsSurfaces(
        "surface_data", ppi=args.search, train=False, transform=transformations,
        pre_transform=SurfacePrecompute(net, args), pre_filter=iface_valid_filter,
        aa=args.aa)
else:
    if args.na=='DNA':
        train_dataset="lists/training_dna.txt"
        test_dataset="lists/testing_dna.txt"
    elif args.na=='RNA':
        train_dataset="lists/training_rna.txt"
        test_dataset="lists/testing_rna.txt"
    elif args.na=='NA':
        train_dataset="lists/training_npi.txt"
        test_dataset="lists/testing_npi.txt"
    
    if args.site:
        prefix=f'site_{args.na.lower()}_'
    elif args.npi:
        prefix=f'npi_{args.na.lower()}_'
    elif args.search:
        prefix=f'search_{args.na.lower()}_'

    full_dataset = NpiDataset('npi_dataset', train_dataset, 
            transform=transformations, pre_transform=SurfacePrecompute(net, args), 
            la=args.la, aa=args.aa, prefix=prefix, pre_filter=iface_valid_filter)
    test_dataset = NpiDataset('npi_dataset', test_dataset, 
            transform=transformations, pre_transform=SurfacePrecompute(net, args), 
            la=args.la, aa=args.aa, prefix=prefix, pre_filter=iface_valid_filter)

# Train/Validation split:
train_nsamples = len(full_dataset)
val_nsamples = int(train_nsamples * args.validation_fraction)
train_nsamples = train_nsamples - val_nsamples
train_dataset, val_dataset = random_split(
    full_dataset, [train_nsamples, val_nsamples]
)
print('Train nsamples:',train_nsamples)
print('Val nsamples:',val_nsamples)
print('Test nsamples:',len(test_dataset))

 
# PyTorch_geometric data loaders:
train_loader = DataLoader(
    train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, follow_batch=batch_vars)
test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)

# Create the model, with a warm restart if applicable:

#optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
optimizer = Lion(net.parameters(), lr=1e-4)
best_loss = 1e10 

starting_epoch = 0
if args.restart_training != "":
    checkpoint = torch.load("models/" + args.restart_training, map_location=args.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    starting_epoch = checkpoint["epoch"]+1
    best_loss = checkpoint["best_loss"]

elif args.transfer_learning != "":
    checkpoint = torch.load("models/" + args.transfer_learning, map_location=args.device)
    for module in checkpoint["model_state_dict"]:
        try:
            net[module].load_state_dict(checkpoint["model_state_dict"][module])
            print('Loaded precomputed module',module)
        except:
            pass 


if not Path("models/").exists():
    Path("models/").mkdir(exist_ok=False)

with open(model_path + '_args.json', 'w') as f:
    json.dump(net_args.__dict__, f, indent=2)

# Training loop (~100 times) over the dataset:
gc.collect()
for i in range(starting_epoch, args.n_epochs):
    # Train first, Test second:
    for dataset_type in ["Train", "Validation", "Test"]:
        if dataset_type == "Train":
            test = False
        else:
            test = True

        suffix = dataset_type
        if dataset_type == "Train":
            dataloader = train_loader
        elif dataset_type == "Validation":
            dataloader = val_loader
        elif dataset_type == "Test":
            dataloader = test_loader

        # Perform one pass through the data:
        info = iterate(
            net,
            dataloader,
            optimizer,
            args,
            test=test,
            epoch_number=i,
        )

        for key, val in info.items():
            if key in [
                "Loss",
                "ROC-AUC",
                "Distance/Positives",
                "Distance/Negatives",
                "Matching ROC-AUC",
            ]:
                print(key ,suffix , i, np.nanmean(val))


        if dataset_type == "Validation":  # Store validation loss for saving the model
            val_loss = np.nanmean(info["Loss"])
        

    if True:  # Additional saves
        if val_loss < best_loss:
            print("Validation loss {}, saving model".format(val_loss))
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                model_path + "_epoch{}".format(i),
            )

            best_loss = val_loss
