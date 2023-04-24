# Standard imports:
import numpy as np
import torch
import json
import os, sys
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from pathlib import Path

# Custom data loader and model:
from data import PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms
from model import dMaSIF
from data_iteration import iterate
from Arguments import parse_inf
from argparse import Namespace

import warnings
warnings.filterwarnings("ignore")

args= parse_inf()

print('Start inference')
print('Arguments:',args)

model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/" + args.experiment_name)

# Ensure reproducability:
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


# Load the train and test datasets:
transformations = (
    Compose([ CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else None
)

# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]

with open(model_path[:model_path.index('_epoch')]+'_args.json', 'r') as f:
    net_args=Namespace(**json.load(f))
print(net_args)

if args.single_pdb != "":
    test_dataset = [load_protein_pair(single_pdb, args.data_dir,single_pdb=True,la=args.la, aa=net_args.aa)]
    test_pdb_ids = [args.single_pdb]
elif args.pdb_list != "":
    with open(args.pdb_list) as f:
        pdb_l = f.read().splitlines()
        test_dataset=[]
        test_pdb_ids=[]
    for pdb in pdb_l:
        try:
            test_dataset.append(load_protein_pair(pdb, args.data_dir,single_pdb=True,la=args.la, aa=net_args.aa))
        except:
            pass
        else:
            test_pdb_ids.append(pdb)
else:
    raise Error
print('Test nsamples:',len(test_dataset))

test_loader = DataLoader(
    test_dataset, batch_size=1, follow_batch=batch_vars, shuffle=False
)


net = dMaSIF(net_args)
net = net.to(args.device)
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)

# Perform one pass through the data:
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