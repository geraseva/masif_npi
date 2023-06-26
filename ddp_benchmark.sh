
python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda5 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:5 cuda:6 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:5 cuda:6 cuda:7 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:4 cuda:5 cuda:6 cuda:7 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:0 cuda:4 cuda:5 cuda:6 cuda:7 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:0 cuda:1 cuda:4 cuda:7 cuda:6 cuda:5 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 1 \
--devices cuda:5 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 2 \
--devices cuda:5 cuda:6 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 3 \
--devices cuda:5 cuda:6 cuda:7 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 4 \
--devices cuda:4 cuda:5 cuda:6 cuda:7 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 5 \
--devices cuda:0 cuda:4 cuda:5 cuda:6 cuda:7 >> ddp.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 6 \
--devices cuda:0 cuda:1 cuda:4 cuda:7 cuda:6 cuda:5 >> ddp.log
