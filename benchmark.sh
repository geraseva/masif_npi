masif_root=$(git rev-parse --show-toplevel)

export PYTHONPATH=${PYTHONPATH}:${masif_root}

#python $masif_root/data_preprocessing/download_pdb.py --pdb_list npidb.list --save_res True 

python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet --dataset ProteinPairsSurfaces # 0.8559492105941394

python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig_v --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V --dataset ProteinPairsSurfaces # 0.8380108538715837

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig_dgcnn --batch_size 64 --embedding_layer DGCNN --site True \
 --single_protein True --random_rotation True --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_MP --dataset ProteinPairsSurfaces # 0.7777782407068814


python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_dgcnn --batch_size 64 --embedding_layer DGCNN --site True \
 --single_protein True --random_rotation True --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet --dataset NpiDataset # 0.7716083364077496

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_atomnet --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet --dataset NpiDataset # 0.9074962983309375

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_atomnetv --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V --dataset NpiDataset # 0.9251051725638855


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
 
