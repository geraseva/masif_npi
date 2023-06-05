import numpy as np
import json
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
#from lion_pytorch import Lion
from pathlib import Path
from argparse import Namespace
import gc
import pykeops
import os
import warnings
from data import *
from model import dMaSIF, Lion
from data_iteration import iterate
import time

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
        args
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
        self.best_loss = 1e10 


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
                epoch_number=epoch,
            )
    
            for key, val in info.items():
                if key in [
                    "Loss",
                    "ROC-AUC",
                    "Distance/Positives",
                    "Distance/Negatives",
                    "Matching ROC-AUC",
                ]:
                    print(key ,suffix , epoch, np.nanmean(val))
    
            if dataset_type == "Validation":  # Store validation loss for saving the model
                val_loss = np.nanmean(info["Loss"])
        
            if val_loss < self.best_loss and self.gpu_id==0:
                print("## Validation loss {}, saving model".format(val_loss))
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": net.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    f"models/{self.args.experiment_name}_epoch{epoch}"
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

    print('# Loading datasets')   
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
    print('## Train nsamples:',train_nsamples)
    print('## Val nsamples:',val_nsamples)
    print('## Test nsamples:',len(test_dataset))

    return (train_dataset,val_dataset,test_dataset), net, optimizer, starting_epoch


def prepare_dataloader(dataset , args):

    batch_vars = ["xyz_p1", "xyz_p2", "atom_xyz_p1", "atom_xyz_p2"]

    train_loader = DataLoader(
        dataset[0], batch_size=args.batch_size, follow_batch=batch_vars,
        shuffle=False, sampler=DistributedSampler(dataset[0]))
    val_loader = DataLoader(
        dataset[1], batch_size=args.batch_size, follow_batch=batch_vars,
        shuffle=False, sampler=DistributedSampler(dataset[1]))
    test_loader = DataLoader(
        dataset[2], batch_size=args.batch_size, follow_batch=batch_vars,
        shuffle=False, sampler=DistributedSampler(dataset[2]))

    return train_loader, val_loader, test_loader


def main(rank: int, rank_list: int, args, dataset, net, optimizer, starting_epoch):

    warnings.simplefilter("ignore")
    ddp_setup(rank, rank_list)
    train_loader, val_loader, test_loader = prepare_dataloader(dataset, args)

    gc.collect()    
    trainer = Trainer(model=net,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      gpu_id=rank_list[rank],
                      args=args)
    trainer.train(starting_epoch)

    destroy_process_group()


if __name__ == "__main__":

    from Arguments import parse_train
    args, net_args = parse_train()

    rank_list=[x for x in args.devices if x!='cpu']

    print(f'# Start {args.mode}')
    print('## Arguments:',args)
    print('## World size:',len(rank_list))

    dataset, net, optimizer, starting_epoch = load_train_objs(args, net_args)
    if not Path("models/").exists():
        Path("models/").mkdir(exist_ok=False)

    with open(f"models/{args.experiment_name}_args.json", 'w') as f:
        json.dump(net_args.__dict__, f, indent=2)
    
    fulltime=time.time()
    mp.spawn(main, args=(rank_list, args, dataset, net, optimizer, starting_epoch), nprocs=len(rank_list))

    fulltime=time.time()-fulltime
    print(f'## Execution complete {fulltime} seconds')


