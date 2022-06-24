masif_root=$(git rev-parse --show-toplevel)

export PYTHONPATH=${PYTHONPATH}:${masif_root}

#python $masif_root/data_preprocessing/download_pdb.py --pdb_list npidb.list --save_res True 

#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export CUDA_LAUNCH_BLOCKING=1 


python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_atomnetv_focal_0 --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --dropout 0.2 --feature_generation AtomNet_V --dataset NpiDataset --n_epochs 10 \
 --focal_loss_gamma 0.0

python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_atomnetv_focal_0.5 --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --dropout 0.2 --feature_generation AtomNet_V --dataset NpiDataset --n_epochs 10 \
 --focal_loss_gamma 0.5

 python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_atomnetv_focal_1 --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --dropout 0.2 --feature_generation AtomNet_V --dataset NpiDataset --n_epochs 10 \
 --focal_loss_gamma 1.0

 python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_atomnetv_focal_2 --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --dropout 0.2 --feature_generation AtomNet_V --dataset NpiDataset --n_epochs 10 \
 --focal_loss_gamma 2.0

 python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_atomnetv_focal_5 --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --dropout 0.2 --feature_generation AtomNet_V --dataset NpiDataset --n_epochs 10 \
 --focal_loss_gamma 5.0
 