Implementation of dMaSIF (https://github.com/FreyrS/dMaSIF) to predict DNA-protein interactions

Make sure that you get all the dependencies from the file requirements.txt

Use these commands to download and unpack the dataset of NA-protein interactions:

    cd masif_npi
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt  \
    --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jMKE2Pae7WNbQv8T1FsJz8UYDeL5CU9T' -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jMKE2Pae7WNbQv8T1FsJz8UYDeL5CU9T" -O npi_dataset.tar.gz && rm -rf /tmp/cookies.txt

    mkdir npi_dataset
    mkdir npi_dataset/raw    
    tar -xzvf npi_dataset.tar.gz -C npi_dataset/raw/

Use this command to run the training benchmark:

    ./benchmark.sh >> log.txt
    
Take into account, that one epoch of training takes about 1 hour. And there are 1000 epochs in the benchmark file in total. 

