
import os, sys
from Arguments import parse_train
args, net_args = parse_train()

if args.device=='cpu':
    os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import torch
import json
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from lion_pytorch import Lion
from pathlib import Path
from argparse import Namespace

import gc

if args.device!='cpu':
    torch.cuda.set_device(args.device)
    if not torch.cuda.is_available():
        args.device='cpu'
        print('Switch to cpu')

from data import *
from model import dMaSIF
from data_iteration import iterate
from helper import *

initialize(device=args.device, seed=args.seed)

print(f'Start {args.mode}')
print('Arguments:',args)

model_path = "models/" + args.experiment_name
if args.mode=='inference':
    with open(model_path[:model_path.index('_epoch')]+'_args.json', 'r') as f:
        net_args=Namespace(**json.load(f))

net = dMaSIF(net_args)
net = net.to(args.device)
starting_epoch = 0

if args.mode=='inference':
    net.load_state_dict(
        torch.load(model_path, map_location=args.device)["model_state_dict"]
    )

elif args.restart_training != "":
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

print('Model loaded')
print('Model arguments:',net_args)


batch_vars = ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2"]

transformations = (
    Compose([CenterPairAtoms(as_single=args.search), 
             RandomRotationPairAtoms(as_single=args.search)])
    if args.random_rotation
    else Compose([])
)

pre_transformations=[SurfacePrecompute(net.preprocess_surface, args),
                     TransferSurface(single_protein=args.single_protein,
                                     threshold=args.threshold )]
if args.search:
    pre_transformations.append(GenerateMatchingLabels(args.threshold))
elif not args.use_surfaces:
    pre_transformations.append(LabelsFromAtoms(single_protein=args.single_protein,
                                               threshold=args.threshold))
if args.single_protein:
    pre_transformations.append(RemoveSecondProtein())
pre_transformations=Compose(pre_transformations)

if args.mode=='train':
    print('Loading datasets')   
    if args.site:
        prefix=f'site_{args.na.lower()}_'
    elif args.npi:
        prefix=f'npi_{args.na.lower()}_'
    elif args.search:
        prefix=f'search_{args.na.lower()}_'
    
    full_dataset = NpiDataset(args.data_dir, args.training_list, use_surfaces=args.use_surfaces,
                transform=transformations, pre_transform=pre_transformations, 
                encoders=args.encoders, prefix=prefix, pre_filter=iface_valid_filter)
    test_dataset = NpiDataset(args.data_dir, args.testing_list, use_surfaces=args.use_surfaces,
                transform=transformations, pre_transform=pre_transformations,
                encoders=args.encoders, prefix=prefix, pre_filter=iface_valid_filter)

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

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, follow_batch=batch_vars, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, follow_batch=batch_vars)
    test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)

else:
    print('Loading testing set')   
    if args.single_pdb != "":
        test_dataset = [load_protein_pair(args.single_pdb, args.data_dir,
            use_surfaces=args.use_surfaces,encoders=args.encoders)]
        test_pdb_ids = [args.single_pdb]
    elif args.pdb_list != "":
        with open(args.pdb_list) as f:
            pdb_l = f.read().splitlines()
        test_dataset=[]
        test_pdb_ids=[]
        for pdb in pdb_l:
            try:
                test_dataset.append(load_protein_pair(pdb, args.data_dir,
                    use_surfaces=args.use_surfaces,encoders=args.encoders))
            except FileNotFoundError:
                print(f'Skipping non-existing files for {pdb}' )
            else:
                test_pdb_ids.append(pdb)

    test_dataset = [pre_transformations(data) for data in tqdm(test_dataset)]
    test_dataset = [transformations(data) for data in tqdm(test_dataset)]
    
    print('Test nsamples:',len(test_dataset))

    test_loader = DataLoader(
        test_dataset, batch_size=1, follow_batch=batch_vars, shuffle=False)

if args.mode=='train':

    print('Start training')

    #optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
    optimizer = Lion(net.parameters(), lr=1e-4)
    best_loss = 1e10 

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

else:
    save_predictions_path = Path("preds/" + args.experiment_name)

    print('Start prediction')

    if not os.path.isdir(save_predictions_path):
        os.makedirs(save_predictions_path)

    info = iterate(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
    )

    info['indexes']=test_pdb_ids

    print('Mean roc-auc:',np.nanmean(info["ROC-AUC"]),'std roc-auc:',np.nanstd(info["ROC-AUC"]))

    for i, pdb in enumerate(info['indexes']):
        print(f"{pdb}: roc-auc {info['ROC-AUC'][i]} Loss {info['Loss'][i]}")