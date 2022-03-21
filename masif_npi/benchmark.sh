

masif_root=$(git rev-parse --show-toplevel)

export PYTHONPATH=${PYTHONPATH}:${masif_root}

#python $masif_root/data_preprocessing/download_pdb.py --pdb_list npidb.list --save_res True 

#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export CUDA_LAUNCH_BLOCKING=1 

python -W ignore::FutureWarning -u ${masif_root}/masif_npi/training.py --device cuda:0 --experiment_name try_npi_3layers_9rad --batch_size 64 --embedding_layer dMaSIF --npi True --single_protein True --random_rotation True --radius 9.0 --n_layers 3
python -W ignore::FutureWarning -u ${masif_root}/masif_npi/training.py --device cuda:0 --experiment_name try_npi_dgcnn --batch_size 64 --embedding_layer DGCNN --npi True --single_protein True --random_rotation True --n_layers 3

