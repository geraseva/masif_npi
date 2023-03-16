
# Standard imports:
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path
import json

# Custom data loader and model:
from data import NpiDataset, PairData, CenterPairAtoms, ProteinPairsSurfaces
from data import RandomRotationPairAtoms
from model import dMaSIF
from data_iteration import iterate, iface_valid_filter, SurfacePrecompute
from helper import *
from Arguments import parser
import pickle
import gc

# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()

print('Start training')
print('Arguments:',args)
model_path = "models/" + args.experiment_name
torch.cuda.set_device(args.device)

if not Path("models/").exists():
    Path("models/").mkdir(exist_ok=False)

with open(model_path + '_args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Ensure reproducibility:
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Create the model, with a warm restart if applicable:
net = dMaSIF(args)
net = net.to(args.device)

# We load the train and test datasets.
# Random transforms, to ensure that no network/baseline overfits on pose parameters:
transformations = (
    Compose([CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else None
)

# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]

if args.na=='DNA':
    train_dataset="lists/training_dna.txt"
    test_dataset="lists/testing_dna.txt"
    la={'DA':1, "DG": 2, "DC":3, "DT":4, '-':0 }
elif args.na=='RNA':
    train_dataset="lists/training_rna.txt"
    test_dataset="lists/testing_rna.txt"
    la={'A':1, "G": 2, "C":3, "U":4, '-':0 }
elif args.na=='NA':
    train_dataset="lists/training_npi.txt"
    test_dataset="lists/testing_npi.txt"
    la={'DA':1, "DG": 2, "DC":3, "DT":4, 'A':1, "G": 2, "C":3, "U":4, '-':0 }

aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "-": 5 }
if args.site:
    la={'-':1 }
    prefix='site_'
elif args.npi:
    prefix='npi_'
elif args.search:
    prefix='search_'
    la={'-':1 }
    aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "P": 5, '-': -1 }

args.aa=aa
# Load the train dataset:
if args.dataset=='NpiDataset':
    full_dataset = NpiDataset('npi_dataset', train_dataset, 
            transform=transformations, pre_transform=SurfacePrecompute(net, args), 
            la=la, aa=aa, prefix=prefix, pre_filter=iface_valid_filter
        )
    test_dataset = NpiDataset('npi_dataset', test_dataset, 
            transform=transformations, pre_transform=SurfacePrecompute(net, args), 
            la=la, aa=aa, prefix=prefix, pre_filter=iface_valid_filter
        )

elif args.dataset=='ProteinPairsSurfaces':
    full_dataset = ProteinPairsSurfaces(
        "surface_data", ppi=args.search, train=True, transform=transformations, 
        pre_transform=SurfacePrecompute(net, args), pre_filter=iface_valid_filter
    )
    test_dataset = ProteinPairsSurfaces(
        "surface_data", ppi=args.search, train=False, transform=transformations,
        pre_transform=SurfacePrecompute(net, args), pre_filter=iface_valid_filter
    )
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


# Baseline optimizer:
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
best_loss = 1e10  # We save the "best model so far"

starting_epoch = 0
if args.restart_training != "":
    checkpoint = torch.load("models/" + args.restart_training, map_location=args.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    starting_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

elif args.transfer_learning != "":
    task=args.transfer_learning[:args.transfer_learning.index('_epoch')]
    args1 = parser.parse_args()
    with open("models/" + task+'_args.json', 'r') as f:
        args1.__dict__ = json.load(f)
    net1 = dMaSIF(args1)
    net1 = net1.to(args.device)
    checkpoint = torch.load("models/" + args.transfer_learning, map_location=args.device)
    net1.load_state_dict(checkpoint["model_state_dict"])
    try:
        net.atomnet.load_state_dict(net1.atomnet.state_dict())
    except:
        pass 
    
    del net1



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
                print(key ,suffix , i, np.mean(val))


        if dataset_type == "Validation":  # Store validation loss for saving the model
            val_loss = np.mean(info["Loss"])
        

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
