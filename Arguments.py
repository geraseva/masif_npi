import argparse
import json

parser = argparse.ArgumentParser(description="Network parameters")

parser.add_argument('--aa', type=json.loads, help='How to encode atoms',default=None)
parser.add_argument(
    "--feature_generation",
    type=str,
    default="AtomNet_V_MP",
    choices=["AtomNet", "AtomNet_MP", "AtomNet_V", "AtomNet_V_MP"],
    help="Which model to use for feature generation",
)
# Surface sampling parameters 
parser.add_argument(
    "--resolution",
    type=float,
    default=1.0,
    help="Resolution of the generated point cloud",
)
parser.add_argument(
    "--distance",
    type=float,
    default=1.05,
    help="Distance parameter in surface generation",
)
parser.add_argument(
    "--variance",
    type=float,
    default=0.1,
    help="Variance parameter in surface generation",
)
parser.add_argument(
    "--sup_sampling", type=int, default=20, help="Sup-sampling ratio around atoms"
)
# Hyper-parameters for the embedding
parser.add_argument(
    "--atom_dims",
    type=int,
    help="Number of atom types",
)
parser.add_argument(
    "--chem_dims",
    type=int,
    default=6,
    help="Number of resulting chemical features",
)
parser.add_argument(
    "--curvature_scales",
    type=list,
    default=[1.0, 2.0, 3.0, 5.0, 10.0],
    help="Scales at which we compute the geometric features (mean and Gauss curvatures)",
)
parser.add_argument(
    "--emb_dims",
    type=int,
    default=8,
    help="Number of input features",
)
parser.add_argument(
    "--in_channels",
    type=int,
    default=16,
    help="Number of embedding dimensions",
)
parser.add_argument(
    "--orientation_units",
    type=int,
    default=14,
    help="Number of hidden units for the orientation score MLP",
)
parser.add_argument(
    "--post_units",
    type=int,
    default=8,
    help="Number of hidden units for the post-processing MLP",
)
parser.add_argument(
    "--n_outputs",
    type=int,
    help="Number of output channels",
)
parser.add_argument(
    "--n_layers", type=int, default=1, help="Number of convolutional layers"
)
parser.add_argument(
    "--radius", type=float, default=9.0, help="Radius to use for the convolution"
)
parser.add_argument('--split', action='store_true', help="Whether to train two conv modules")
parser.add_argument("--dropout", type=float, default=0.0,
    help="Amount of Dropout for the input features"
)


train_inf_parser=argparse.ArgumentParser(add_help=False)

train_inf_parser.add_argument(
    '-e',"--experiment_name", type=str, help="Name of experiment", required=True
)
train_inf_parser.add_argument(
    "--na",
    type=str,
    default='NA',
    choices=["DNA", "RNA", 'NA', 'protein'],
    help="Which dataset to use for training",
)
train_inf_parser.add_argument(
    "--loss",
    type=str,
    default=None,
    choices=["CELoss", "BCELoss", "FocalLoss"],
    help="Which loss function to use",
)
train_inf_parser.add_argument(
    "--focal_loss_gamma",
    type=float,
    default=2,
    help="Gamma parameter for focal loss",
)
train_inf_parser.add_argument("--seed", type=int, default=42, help="Random seed")
train_inf_parser.add_argument( "--random_rotation", type=bool, default=True, 
    help="Move proteins to center and add random rotation",
)
train_inf_parser.add_argument( "--single_protein", type=bool, 
    help="Use single protein in a pair or both")

task = train_inf_parser.add_mutually_exclusive_group(required=True)
task.add_argument("--site", action='store_true', 
    help="Predict interaction site")
task.add_argument("--npi", action='store_true', 
    help="Predict nucleotide binding")
task.add_argument( "--search", action='store_true', 
    help="Predict matching between two partners",)

train_inf_parser.add_argument(
    "--device", type=str, default="cuda:0", help="Which gpu/cpu to train on"
)
train_inf_parser.add_argument( "--threshold", type=float, default=None,
    help="Distance threshold for interaction")
train_inf_parser.add_argument('--la', type=json.loads, help='How to encode residue labels',
    default=None)
train_inf_parser.add_argument('--aa', type=json.loads, help='How to encode atoms',default=None)


train_parser = argparse.ArgumentParser(description="Training parameters", parents=[train_inf_parser], prog='training')

train_parser.add_argument(
    "--n_epochs", type=int, default=50, help="Number of training epochs"
)
train_parser.add_argument(
    "--restart_training",
    type=str,
    default="",
    help="Which model to restart the training from",
)
train_parser.add_argument(
    "--transfer_learning",
    type=str,
    default="",
    help="Which model to use for parameters transfer",
)
train_parser.add_argument(
    "--validation_fraction",
    type=float,
    default=0.1,
    help="Fraction of training dataset to use for validation",
)

inf_parser = argparse.ArgumentParser(description="Inference parameters", parents=[train_inf_parser], prog='inference')
set_group = inf_parser.add_mutually_exclusive_group(required=True)
set_group.add_argument(
    "--single_pdb",
    type=str,
    default="",
    help="Which structure to do inference on",
)
set_group.add_argument(
    "--pdb_list",
    type=str,
    default="",
    help="Which structures to do inference on",
)

inf_parser.add_argument(
    "--data_dir", type=str, required=True, help="From where to take numpy data"
)

def parse_train():
    args, _ = train_parser.parse_known_args()
    net_args, _ = parser.parse_known_args()
    if args.la==None:
        if args.npi:
            if args.na=='DNA':
                args.la={'DA':1, "DG": 2, "DC":3, "DT":4, '-':0 }
            elif args.na=='RNA':
                args.la={'A':1, "G": 2, "C":3, "U":4, '-':0 }
            elif args.na=='NA':
               args.la={'DA':1, "DG": 2, "DC":3, "DT":4, 'A':1, "G": 2, "C":3, "U":4, '-':0 }
        else:
            args.la={'-':1 }
    if net_args.aa==None:
        if args.search and args.na!='protein': 
            aa={"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "P": 5, '-': 4 }
        else:
            aa={"C": 0, "H": 1, "O": 2, "N": 3, "-": 4}    
    args.aa=aa
    net_args.aa=aa
    if net_args.atom_dims==None:
        net_args.atom_dims=max(net_args.aa.values())+1
    if net_args.n_outputs==None:
        if args.npi:
            net_args.n_outputs=max(args.la.values())+1
        elif args.site:
            net_args.n_outputs=1
        elif args.search:
            net_args.n_outputs=0
    if args.threshold==None:
        if args.search:
            if args.na=='protein':
                args.threshold=1 # distance between two surface points
            else:
                args.threshold=2
        else:
            args.threshold=5 # distance between surface point and atom center
    if args.single_protein==None:
        if args.search:
            args.single_protein=False
        else:
            args.single_protein=True
    if args.loss==None:  
        if args.npi:
            args.loss='FocalLoss'
        else:
            args.loss='BCELoss'  

    if args.search:
        net_args.split=True
    else:
        net_args.split=False


    return args, net_args

def parse_inf():
    inf_parser.set_defaults(random_rotation=False, device='cpu')
    args, _ = inf_parser.parse_known_args()
    if args.la==None:
        if args.npi:
            if args.na=='DNA':
                args.la={'DA':1, "DG": 2, "DC":3, "DT":4, '-':0 }
            elif args.na=='RNA':
                args.la={'A':1, "G": 2, "C":3, "U":4, '-':0 }
            elif args.na=='NA':
               args.la={'DA':1, "DG": 2, "DC":3, "DT":4, 'A':1, "G": 2, "C":3, "U":4, '-':0 }
        else:
            args.la={'-':1 }
    if args.threshold==None:
        if args.search: 
            if args.na=='protein':
                args.threshold=1
            else:
                args.threshold=2
        else:
            args.threshold=5 # distance between surface point and atom center
    if args.single_protein==None:
        if args.search:
            args.single_protein=False
        else:
            args.single_protein=True
    if args.loss==None:  
        if args.npi:
            args.loss='FocalLoss'
        else:
            args.loss='BCELoss' 
 
    return args