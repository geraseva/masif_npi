masif_root=$(git rev-parse --show-toplevel)

export PYTHONPATH=${PYTHONPATH}:${masif_root}

#python $masif_root/data_preprocessing/download_pdb.py --pdb_list lists/dnaprot.list --save_res True 

# experiments on the protein-protein interaction dataset

python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet --dataset ProteinPairsSurfaces 

python -W ignore::FutureWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig_v --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V --dataset ProteinPairsSurfaces 

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig_mp --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_MP --dataset ProteinPairsSurfaces 

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name orig_v_mp --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V_MP --dataset ProteinPairsSurfaces 

# binary prediction of dna-protein interaction site

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_dna_atomnet --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet --dataset NpiDataset 
 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_dna_atomnet_v --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V --dataset NpiDataset 

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_dna_atomnet_mp --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_MP --dataset NpiDataset 

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_dna_atomnet_v_mp --batch_size 64 --embedding_layer dMaSIF --site True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V_MP --dataset NpiDataset 

# binary prediction of rna-protein interaction site

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_rna_atomnet --batch_size 64 --embedding_layer dMaSIF --site True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet --dataset NpiDataset 

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_rna_atomnet_v --batch_size 64 --embedding_layer dMaSIF --site True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V --dataset NpiDataset 

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_rna_atomnet_mp --batch_size 64 --embedding_layer dMaSIF --site True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_MP --dataset NpiDataset 

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name binary_rna_atomnet_v_mp --batch_size 64 --embedding_layer dMaSIF --site True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss BCELoss \
 --feature_generation AtomNet_V_MP --dataset NpiDataset 

# prediction of nucleotides in dna-protein interaction site

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_dna_atomnet --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet --dataset NpiDataset 

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_dna_atomnet_v --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet_V --dataset NpiDataset 

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_dna_atomnet_mp --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet_MP --dataset NpiDataset 

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_dna_atomnet_v_mp --batch_size 64 --embedding_layer dMaSIF --npi True \
 --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet_V_MP --dataset NpiDataset 

# prediction of nucleotides in rna-protein interaction site

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_rna_atomnet --batch_size 64 --embedding_layer dMaSIF --npi True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet --dataset NpiDataset 
 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_rna_atomnet_v --batch_size 64 --embedding_layer dMaSIF --npi True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet_V --dataset NpiDataset 

python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_rna_atomnet_mp --batch_size 64 --embedding_layer dMaSIF --npi True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet_MP --dataset NpiDataset 

 python -W ignore::UserWarning -u ${masif_root}/training.py --device cuda:0 \
 --experiment_name npi_rna_atomnet_v_mp --batch_size 64 --embedding_layer dMaSIF --npi True \
 --na RNA --single_protein True --random_rotation True --radius 9.0 --n_layers 3 --loss FocalLoss \
 --feature_generation AtomNet_V_MP --dataset NpiDataset 