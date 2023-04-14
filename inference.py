# Standard imports:
import numpy as np
import torch
import json
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path

# Custom data loader and model:
from data import PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms
from model import dMaSIF
from data_iteration import iterate
from Arguments import parser

args = parser.parse_args()
model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/" + args.experiment_name)
single_pdb=args.single_pdb
pdb_list=args.pdb_list
device=args.device

with open(model_path[:model_path.index('_epoch')]+'_args.json', 'r') as f:
    args.__dict__ = json.load(f)
args.device=device

# Ensure reproducability:
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
if args.na=='DNA':
    la={'DA':1, "DG": 2, "DC":3, "DT":4, '-':0 }
elif args.na=='RNA':
    la={'A':1, "G": 2, "C":3, "U":4, '-':0 }
elif args.na=='NA':
    la={'DA':1, "DG": 2, "DC":3, "DT":4, 'A':1, "G": 2, "C":3, "U":4, '-':0 }

aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "-": 5 }
if args.site:
    la={'-':1 }
elif args.search:
    la={'-':1 }
    if args.dataset=='NpiDataset':
        aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "P": 5, '-': -1 }

if args.dataset=='NpiDataset':
    single_data_dir = "./npi_dataset/raw"
elif args.dataset=='ProteinPairsSurfaces':
    single_data_dir = "./surface_data/raw/01-benchmark_surfaces_npy"
    aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, 'Se':4, "-": 5 }
    la=None

args.aa=aa

if single_pdb != "":
    test_dataset = [load_protein_pair(single_pdb, single_data_dir,single_pdb=True,la=la, aa=aa)]
    test_pdb_ids = [single_pdb]
elif pdb_list != "":
    with open(pdb_list) as f:
        pdb_l = f.read().splitlines()
    test_dataset = [load_protein_pair(pdb, single_data_dir,single_pdb=True,la=la, aa=aa) for pdb in pdb_l]
    test_pdb_ids = [pdb for pdb in pdb_l]
else:
    raise Error


# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
test_loader = DataLoader(
    test_dataset, batch_size=1, follow_batch=batch_vars, shuffle=False
)

net = dMaSIF(args)
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)

net = net.to(args.device)

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

print('Mean roc-auc:',np.mean(info["ROC-AUC"]),'std roc-auc:',np.std(info["ROC-AUC"]))

for i, pdb in enumerate(info['indexes']):
    print(f"{pdb}: roc-auc {info['ROC-AUC'][i]} Loss {info['Loss'][i]}")
