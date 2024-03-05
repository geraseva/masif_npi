Implementation of dMaSIF (https://github.com/FreyrS/dMaSIF) to predict NA-protein interactions

Required libraries:
- reduce 
- biopython
- pytorch
- pykeops
- pymol (optional)

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