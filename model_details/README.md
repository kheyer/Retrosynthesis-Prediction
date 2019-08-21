# Model Details

This directory contains information on training transformer sequence 2 sequence models for Retrosynthesis reaction prediction using [OpenNMT](https://github.com/OpenNMT/OpenNMT-py)

## Setting up OpenNMT

Clone the [OpenNMT repo](https://github.com/OpenNMT/OpenNMT-py)

Follow the setup instructions in the repo. I found it best to create a new anaconda environment for it.

## Dataset Generation

Datasets can be downloaded [here](https://www.dropbox.com/s/ze4bdif8sqjx5jx/Retrosynthesis%20Data.zip?dl=0)

Datasets should be plcaed in the `data` directory in the `OpenNMT-py` directory using a different data folder for each version of the dataset (no augmentation, 4x, 16x and 40x augmentation)

In the `OpenNMT-py` directory, run the following command:

    python preprocess.py -train_src data/dataset_directory/train_source.txt \
    -train_tgt data/dataset_directory/train_targets.txt -valid_src data/dataset_directory/valid_source.txt \
    -valid_tgt data/dataset_directory/valid_targets.txt -save_data data/dataset_directory/dataset_name \
    -src_seq_length 1000 -tgt_seq_length 1000 -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
    
## Model Training

Move the `model_config.yml` file in this directory to the `config` directory in the `OpenNMT-py` directory.

Update the `data` and `save_model` fields for the dataset created above

In the `OpenNMT-py` directory, run the following command:

    python train.py -config config/model_config.yml
    
## Model Prediction
