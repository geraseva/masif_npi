Make sure that you get all the dependencies from the file requirements.txt

Use these commands to download and unpack the dataset:

    cd masif_npi
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt  \
    --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SCZZ9GHHpgfJwmXDLYZrfRYXY2eojVKK' -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SCZZ9GHHpgfJwmXDLYZrfRYXY2eojVKK" -O npi_dataset.tar.gz && rm -rf /tmp/cookies.txt

    mkdir npi_dataset
    mkdir npi_dataset/raw
    tar -xzvf npi_dataset.tar.gz -C npi_dataset/raw/

Use this command to run the benchmark:

    ./benchmark.sh >> log.txt
    
Take into account, that one epoch of training takes about 45 minutes. And there are 350 epochs in the benchmark file in total.

