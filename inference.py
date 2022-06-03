# Standard imports:
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path

# Custom data loader and model:
from data import PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
from data_iteration import iterate
from helper import *
from Arguments import parser

args = parser.parse_args()
model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/" + args.experiment_name)
single_pdb=args.single_pdb
pdb_list=args.pdb_list

with open(model_path[:model_path.index('_epoch')]+'_args.json', 'r') as f:
    args.__dict__ = json.load(f)

# Ensure reproducability:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


# Load the train and test datasets:
transformations = (
    Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else Compose([NormalizeChemFeatures()])
)
if args.site:
    la={'-':1 }
else:
    la={'DA':1, "DG": 2, "DC":3, "DT":4, '-':0 }

aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "-": 5}

if single_pdb != "":
    single_data_dir = "./data_preprocessing/npys/"
    test_dataset = [load_protein_pair(single_pdb, single_data_dir,single_pdb=True,la=la, aa=aa)]
    test_pdb_ids = [single_pdb]
elif pdb_list != "":
    with open(pdb_list) as f:
        pdb_l = f.read().splitlines()
    single_data_dir = "./masif_npi/npys/"
    test_dataset = [load_protein_pair(pdb, single_data_dir,single_pdb=True,la=la, aa=aa) for pdb in pdb_l]
    test_pdb_ids = [pdb for pdb in pdb_l]
else:
    raise Error


# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
test_loader = DataLoader(
    test_dataset, batch_size=1, follow_batch=batch_vars
)

net = dMaSIF(args)
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)
net = net.to(args.device)

# Perform one pass through the data:
info = iterate(
    net,
    test_loader,
    None,
    args,
    test=True,
    save_path=save_predictions_path,
    pdb_ids=test_pdb_ids,
    roccurve=True
)
np.save(f"preds/{args.experiment_name}_roc.npy", info["ROC_curve"])
print(zip(test_pdb_ids,info["ROC-AUC"]))
print(np.mean(info["ROC-AUC"]),np.std(info["ROC-AUC"]))

#np.save(f"timings/{args.experiment_name}_convtime.npy", info["conv_time"])
#np.save(f"timings/{args.experiment_name}_memoryusage.npy", info["memory_usage"])
