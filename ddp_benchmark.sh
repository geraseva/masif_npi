
#python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
#--devices cuda5 >> yoba.log

#python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
#--devices cuda:5 cuda:6 >> yoba.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 8 \
--devices cuda:0 cuda:1 cuda:4 cuda:7 cuda:6 cuda:5 >> yoba.log

#python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 1 \
#--devices cuda:5 >> yoba.log

#python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 2 \
#--devices cuda:5 cuda:6 >> yoba.log

python ddp_training.py train -e yoba --search --na protein --n_epochs 1 --batch_size 6 \
--devices cuda:0 cuda:1 cuda:4 cuda:7 cuda:6 cuda:5 >> yoba.log
