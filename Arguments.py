import argparse
import json

net_parser = argparse.ArgumentParser(description="Network parameters", add_help=False,usage='')

net_parser.add_argument(
    "--feature_generation",
    type=str,
    default="AtomNet_V_MP",
    choices=["AtomNet", "AtomNet_MP", "AtomNet_V", "AtomNet_V_MP"],
    help="Which model to use for feature generation",
)
# Surface sampling parameters 
net_parser.add_argument(
    "--resolution",
    type=float,
    default=1.0,
    help="Resolution of the generated point cloud",
)
net_parser.add_argument(
    "--distance",
    type=float,
    default=None,
    help="Distance parameter in surface generation",
)
net_parser.add_argument(
    "--variance",
    type=float,
    default=0.1,
    help="Variance parameter in surface generation",
)
net_parser.add_argument(
    "--sup_sampling", type=int, default=None, help="Sup-sampling ratio around atoms"
)
# Hyper-parameters for the embedding
net_parser.add_argument(
    "--atom_dims",
    type=int,
    help="Number of atom types",
)
net_parser.add_argument(
    "--chem_dims",
    type=int,
    default=6,
    help="Number of resulting chemical features",
)
net_parser.add_argument(
    "--knn",
    type=int,
    default=None,
    help="Number of nearest atoms",
)
net_parser.add_argument(
    "--curvature_scales",
    type=list,
    default=[1.0, 2.0, 3.0, 5.0, 10.0],
    help="Scales at which we compute the geometric features (mean and Gauss curvatures)",
)
net_parser.add_argument(
    "--emb_dims",
    type=int,
    default=8,
    help="Number of embedding dimensions",
)

net_parser.add_argument(
    "--orientation_units",
    type=int,
    default=14,
    help="Number of hidden units for the orientation score MLP",
)
net_parser.add_argument(
    "--post_units",
    type=int,
    default=8,
    help="Number of hidden units for the post-processing MLP",
)
net_parser.add_argument(
    "--n_outputs",
    type=int,
    help="Number of output channels",
)
net_parser.add_argument(
    "--n_layers", type=int, default=3, help="Number of convolutional layers"
)
net_parser.add_argument(
    "--radius", type=float, default=9.0, help="Radius to use for the convolution"
)
net_parser.add_argument('--split', action='store_true', help="Whether to train two conv modules")
net_parser.add_argument("--dropout", type=float, default=0.0,
    help="Amount of Dropout for the input features"
)
net_parser.add_argument('--encoders', type=json.loads, 
    help='How to encode atom labels', default={})

train_inf_parser=argparse.ArgumentParser(add_help=False)

required = train_inf_parser.add_argument_group('required arguments')

required.add_argument(
    '-e',"--experiment_name", type=str, help="Name of experiment", required=True
)

task = required.add_mutually_exclusive_group(required=True)

task.add_argument("--site", action='store_true', 
    help="Predict interaction site")
task.add_argument("--npi", action='store_true', 
    help="Predict nucleotide binding")
task.add_argument( "--search", action='store_true', 
    help="Predict matching between two partners")

required.add_argument(
    "--na",
    type=str,
    choices=["DNA", "RNA", 'NA', 'protein'],
    help="Which dataset to use for training",
    required=True
)
train_inf_parser.add_argument(
    "--loss",
    type=str,
    default=None,
    choices=["CELoss", "BCELoss", "FocalLoss"],
    help="Which loss function to use",
)
train_inf_parser.add_argument("--seed", type=int, default=42, help="Random seed")

train_inf_parser.add_argument( "--single_protein", help="Use single protein in a pair or both",
                              type=lambda x: (str(x).lower() == 'true'))
train_inf_parser.add_argument( "--random_rotation", type=lambda x: (str(x).lower() == 'true'),
    help="Move proteins to center and add random rotation", default=None)
train_inf_parser.add_argument(
    "--data_dir", type=str, help="Where pdb files are stored"
)
train_inf_parser.add_argument(
    "--device", type=str, default=None, help="Which gpu/cpu to train on"
)
train_inf_parser.add_argument(
    "--devices", type=str, default=None, nargs='*',
    help="Which gpus to train on"
)
train_inf_parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size"
)
train_inf_parser.add_argument( "--threshold", type=float, 
    default=None, help="Distance threshold for interaction")
train_inf_parser.add_argument(
    "--no_h",
    action='store_true',
    help="Whether to remove hydrogens",
)
train_inf_parser.add_argument('--encoders', type=json.loads, 
    help='How to encode atom labels', default={})


main_parser=argparse.ArgumentParser(prog='train_inf')

subparsers = main_parser.add_subparsers(title='mode',help='To train or to validate', required=True, dest='mode')

train_parser=subparsers.add_parser('train', description="Training parameters", parents=[train_inf_parser],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog=f'network arguments:{net_parser.format_help().split("optional arguments:")[1]}')

train_parser._action_groups=train_parser._action_groups[::-1]

train_parser.add_argument(
    "--training_list",
    type=str,
    default=None,
    help="Which structures to train on",
)
train_parser.add_argument(
    "--testing_list",
    type=str,
    default=None,
    help="Which structures to test on",
)

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

inf_parser=subparsers.add_parser('inference',description="Inference parameters", parents=[train_inf_parser])
inf_parser.add_argument(
    "--protonate",
    action='store_true',
    help="Whether to protonate pdb files",
)
inf_parser.add_argument(
    "--out_dir",
    default=None,
    type=str, help="Where to store output npy files"
)
set_group = inf_parser._action_groups[-1].add_mutually_exclusive_group(required=True)
inf_parser._action_groups=inf_parser._action_groups[::-1]
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


def parse_train():
    args, _ = main_parser.parse_known_args()

    if args.encoders=={}:
        if args.npi:
            if args.na=='DNA':
                args.encoders['atom_resnames']=[{'name': 'atom_res',
                                                 'encoder': {'DA':1, "DG": 2, "DC":3, "DT":4, '-':0}
                                                 }]
            elif args.na=='RNA':
                args.encoders['atom_resnames']=[{'name': 'atom_res',
                                                 'encoder': {'A':1, "G": 2, "C":3, "U":4, '-':0 }
                                                 }]
            elif args.na=='NA':
                args.encoders['atom_resnames']=[{'name': 'atom_res',
                                                 'encoder': {'DA':1, "DG": 2, "DC":3, "DT":4, 
                                                 'A':1, "G": 2, "C":3, "U":4, '-':0 }
                                                 }]
        else:
            args.encoders['atom_resnames']=[{'name': 'atom_res',
                                                 'encoder': {'-':1 }
                                            }]
        if args.search and args.na!='protein': 
            args.encoders['atom_types']=[{'name': 'atom_types',
                                                 'encoder': {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "P": 5, '-': 4 }},
                                            {'name': 'atom_rad',
                                                 'encoder': {'H': 110, 'C': 170, 'N': 155, 'O': 152, '-': 180}
                                            }]
        else:
            args.encoders['atom_types']=[{'name': 'atom_types',
                                                 'encoder': {"C": 0, "H": 1, "O": 2, "N": 3, '-': 4 }},
                                            {'name': 'atom_rad',
                                                 'encoder': {'H': 110, 'C': 170, 'N': 155, 'O': 152, '-': 180}
                                            }]
        if args.no_h:
            for encoder in args.encoders['atom_types']:
                val=encoder['encoder'].pop('H')
                if encoder['name']=='atom_rad':
                    continue
                for key in encoder['encoder']:
                    if encoder['encoder'][key]>val:
                        encoder['encoder'][key]-=1 
            args.encoders['atom_types'].append({'name': 'mask',
                                                   'encoder': {"H": 0, "-": 1}})
    if args.threshold==None:
        if args.search:
            args.threshold=1 # distance between two surface points
        else:
            args.threshold=5 # distance between surface point and atom center
    if args.single_protein==None:
        if args.search:
            args.single_protein=False
        else:
            args.single_protein=True
        if args.devices==None:
            if args.device==None:
                args.device='cuda:0'
            args.devices=[args.device]
        elif args.device==None:
            args.device=args.devices[0]
        if args.loss==None:  
            if args.npi:
                args.loss='FocalLoss'
            else:
                args.loss='BCELoss'  
            
    if args.mode=='train':
        net_args, _ = net_parser.parse_known_args()
        if args.random_rotation==None:
            args.random_rotation=True

        if net_args.atom_dims==None:
            if args.search and args.na!='protein': 
                net_args.atom_dims=6
            else:
                net_args.atom_dims=5
            if args.no_h:
                net_args.atom_dims-=1
        if net_args.n_outputs==None:
            if args.npi:
                net_args.n_outputs=5
            elif args.site:
                net_args.n_outputs=1
            elif args.search:
                net_args.n_outputs=0
        if net_args.encoders=={}:
            net_args.encoders=args.encoders
        if args.search:
            net_args.split=True
        else:
            net_args.split=False
        if net_args.distance==None:
            if args.no_h:
                net_args.distance=1.25
            else:
                net_args.distance=1.05
        if net_args.sup_sampling==None:
            if args.no_h:
                net_args.sup_sampling=34
            else:
                net_args.sup_sampling=20
        if net_args.knn==None:
            if args.no_h:
                net_args.knn=10
            else:
                net_args.knn=16

        if args.training_list==None:
            if args.na=='protein':
                args.training_list='lists/training_ppi.txt'
            elif args.na=='DNA':
                args.training_list="lists/training_dna.txt"
            elif args.na=='RNA':
                args.training_list="lists/training_rna.txt"
            elif args.na=='NA':
                args.training_list="lists/training_npi.txt"

        if args.testing_list==None:
            if args.na=='protein':
                args.testing_list="lists/testing_ppi.txt"
            elif args.na=='DNA':
                args.testing_list="lists/testing_dna.txt"
            elif args.na=='RNA':
                args.testing_list="lists/testing_rna.txt"
            elif args.na=='NA':
                args.testing_list="lists/testing_npi.txt"
        if args.data_dir==None:
            args.data_dir='datasets/'
        net_args=net_args.__dict__
    else:
        net_args=None
        if args.random_rotation==None:
            args.random_rotation=False
        if args.data_dir==None:
            args.data_dir='datasets/raw'
        if args.out_dir is None:
            args.out_dir='preds/'+args.experiment_name    
    
   
    return args.__dict__, net_args
