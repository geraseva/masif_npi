export PYTHONPATH=${PYTHONPATH}:$(git rev-parse --show-toplevel)

# binary prediction of PPI site

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig --site \
 --feature_generation AtomNet --na protein >> logs/log_orig.txt 

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_v --site \
 --feature_generation AtomNet_V --na protein >> logs/log_orig.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_mp --site \
 --feature_generation AtomNet_MP --na protein >> logs/log_orig.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_v_mp --site \
 --feature_generation AtomNet_V_MP --na protein >> logs/log_orig.txt

# binary prediction of dna-protein interaction site

python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_dna_atomnet --site \
 --feature_generation AtomNet --na DNA >> logs/log_dna_site.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_dna_atomnet_v --site \
 --feature_generation AtomNet_V --na DNA >> logs/log_dna_site.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_dna_atomnet_mp --site \
 --feature_generation AtomNet_MP --na DNA >> logs/log_dna_site.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_dna_atomnet_v_mp --site \
 --feature_generation AtomNet_V_MP --na DNA >> logs/log_dna_site.txt

# binary prediction of rna-protein interaction site

 python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_rna_atomnet --site \
 --na RNA --feature_generation AtomNet >> logs/log_rna_site.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_rna_atomnet_v --site \
 --na RNA --feature_generation AtomNet_V >> logs/log_rna_site.txt 

python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_rna_atomnet_mp --site \
 --na RNA --feature_generation AtomNet_MP >> logs/log_rna_site.txt 

 python3 train_inf.py train --device cuda:0 \
 --experiment_name binary_rna_atomnet_v_mp --site \
 --na RNA --feature_generation AtomNet_V_MP >> logs/log_rna_site.txt 

# prediction of nucleotides in dna-protein interaction site

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_dna_atomnet --npi \
 --feature_generation AtomNet --na DNA >> logs/log_dna_npi.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_dna_atomnet_v --npi \
 --feature_generation AtomNet_V --na DNA >> logs/log_dna_npi.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_dna_atomnet_mp --npi \
 --feature_generation AtomNet_MP --na DNA >> logs/log_dna_npi.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_dna_atomnet_v_mp --npi \
 --feature_generation AtomNet_V_MP --na DNA >> logs/log_dna_npi.txt

# prediction of nucleotides in rna-protein interaction site

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_rna_atomnet --npi \
 --na RNA --feature_generation AtomNet >> logs/log_rna_npi.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_rna_atomnet_v --npi \
 --na RNA --feature_generation AtomNet_V >> logs/log_rna_npi.txt 

python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_rna_atomnet_mp --npi \
 --na RNA --feature_generation AtomNet_MP >> logs/log_rna_npi.txt 

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_rna_atomnet_v_mp --npi \
 --na RNA --feature_generation AtomNet_V_MP >> logs/log_rna_npi.txt 

 # prediction of interactions on PPI dataset

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_search_atomnet --search \
 --feature_generation AtomNet --na protein >> logs/log_orig_search.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_search_atomnet_v --search \
 --feature_generation AtomNet_V --na protein >> logs/log_orig_search.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_search_atomnet_mp --search \
 --feature_generation AtomNet_MP --na protein >> logs/log_orig_search.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name orig_search_atomnet_v_mp --search \
 --feature_generation AtomNet_V_MP --na protein >> logs/log_orig_search.txt

 # prediction of interactions on NPI dataset

python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_search_atomnet --search \
 --feature_generation AtomNet --na NA >> logs/log_na_search.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_search_atomnet_v --search \
 --feature_generation AtomNet_V --na NA >> logs/log_na_search.txt

python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_search_atomnet_mp --search \
 --feature_generation AtomNet_MP --na NA >> logs/log_na_search.txt

 python3 train_inf.py train --device cuda:0 \
 --experiment_name npi_search_atomnet_v_mp --search \
 --feature_generation AtomNet_V_MP --na NA >> logs/log_na_search.txt

 # time consumptions measurements

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda5 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:5 cuda:6 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:5 cuda:6 cuda:7 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:4 cuda:5 cuda:6 cuda:7 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:0 cuda:4 cuda:5 cuda:6 cuda:7 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:0 cuda:1 cuda:4 cuda:7 cuda:6 cuda:5 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 1 \
--devices cuda:5 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 2 \
--devices cuda:5 cuda:6 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 3 \
--devices cuda:5 cuda:6 cuda:7 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 4 \
--devices cuda:4 cuda:5 cuda:6 cuda:7 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 5 \
--devices cuda:0 cuda:4 cuda:5 cuda:6 cuda:7 >> logs/ddp.log

python3 train_inf.py train -e _ --search --na protein --n_epochs 1 --batch_size 6 \
--devices cuda:0 cuda:1 cuda:4 cuda:7 cuda:6 cuda:5 >> logs/ddp.log