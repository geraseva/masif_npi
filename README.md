Implementation of dMaSIF (https://github.com/FreyrS/dMaSIF) to predict NA-protein interactions

Required libraries:
- reduce 
- biopython
- pytorch
- pytorch geometric
- pykeops
- pymol (optional)

Use these commands to download and unpack the dataset of NA-protein interactions:
```
cd masif_npi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt  \
--keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1q8DU3kfeTORHaylOhQ4QVA4wPU1Jv4rQ' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1q8DU3kfeTORHaylOhQ4QVA4wPU1Jv4rQ" -O pdbs.tar.gz && rm -rf /tmp/cookies.txt

mkdir datasets
tar -xzvf pdbs.tar.gz -C datasets/
```
Commands used for training are in the `benchmark.sh` file.
    
Inference can be performed using commands like:
```
python3 train_inf.py inference --device cuda:0 --batch_size 4 \
--experiment_name npi_search_b2 --search  --na NA \
--pdb_list lists/testing_dna.txt 

python3 train_inf.py inference --device cpu --batch_size 1 \
--experiment_name npi_site_b2  --site --na RNA \
--data_dir pdbs --single_pdb "7did.pdb A C" --protonate \
--out_dir npys/
```
Make sure that model weights exist in folder `models/` as well as argument `.json` files.

[More details](https://doi.org/10.1007/978-3-031-49435-2_17) about the work. 