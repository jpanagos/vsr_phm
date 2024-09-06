# Visual Speech Recognition Using Compact Hypercomplex Neural Networks

This is the official repository for the paper _Visual Speech Recognition Using Compact Hypercomplex Neural Networks_.

https://www.sciencedirect.com/science/article/pii/S0167865524002587

which is an extended version of our previous work: _Compressing Audio Visual Speech Recognition Models With Parameterized Hypercomplex Layers_ ([paper](https://doi.org/10.1145/3549737.3549785))([code](https://github.com/jpanagos/speech_recognition_with_hypercomplex_layers)), published as a conference paper in SETN 2022. 

Instructions on how to train and test models presented in the article are provided below.

## Setup

All experiments were performed in a **conda** environment, with **Python 3.7**, although newer versions should also work.

Create an environment with conda, activate it, and then install requirements with pip:

```pip install -r requirements.txt```

> In case `pip` fails due to wrong package installation order, you will have to install _some_ (or all) packages manually.

## Training

To train on [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) you first have to accept its license agreement; follow the instructions on the [link](https://www.bbc.co.uk/rd/projects/lip-reading-datasets).

After downloading and extracting the archive, pre-process the data following the instructions in the **preprocessing** directory:

📁[preprocessing](https://github.com/jpanagos/vsr_phm/tree/master/preprocessing)

### Additional training options

The training process can be parameterized with the following options:
```
--annonation-direc              path of LRW raw files
--data-dir                      path of pre-processed LRW files
--config-path                   path for the config file that specifies the network architectures
--epochs                        specify amount of epochs (default = 80)
--optimizer                     set optimizer (default = 'adamw', other options are 'sgd' and 'adam')
--lr,                           set learning rate (default = 3e-4)
--batch_size                    set batch size (default = 32)
--workers                       set number of workers for data loading
--interval                      set logger display interval
--init-epoch                    used for resuming training from a checkpoint
--test                          used for testing using a checkpoint
--model-path                    path to checkpoint file (for resuming training, or testing)
--alpha                         set mixup alpha (default = 0.4)
--expn                          path to save files (training log and weights) to
# Hypercomplex-specific settings
--h                             enables hypercomplex layers in the entire network except front-end
--f                             enables hypercomplex layers in the front-end
--n                             used with --h, --f, controls the hyper-parameter n
```

### Examples

To _train_ a model for 40 epochs with N=16:

`python main.py --config-path <path_to_config_file> --data-dir <path_of_preprocessing> --annonation-dir <lrw_path> --epochs 40 --h --n 16 --expn <output_folder>`

And for a model with N=2, and PHM layers in the 3d front-end layer, using the SGD optimizer with a different learning rate:

`python main.py --config-path <path_to_config_file> --data-dir <path_of_preprocessing> --annonation-dir <lrw_path> --optimizer sgd --lr 0.005 --epochs 40 --h --n 2 --f --expn <output_folder>`

>- `<config-path>` is a path to a JSON config file specifying the network architecture **(file)**
>- `<data-dir>` is where the preprocessed files were saved **(directory)**
>- `<annonation-dir>` is where the raw LRW files were unzipped **(directory)**
>- `<output_folder>` is where all files will be saved (make sure you have permissions) **(directory)**

To _resume training_ from a checkpoint (e.g., with N=4, trained for 50 epochs):

`python main.py --config-path <path_to_config_file> --data-dir <path_of_preprocessing> --annonation-dir <lrw_path> --init-epoch 50 --epochs 100 --h --n 4 --model-path <path_to_checkpoint_file> --expn <output_folder>`

> To resume training from a checkpoint, make sure that **a)** the config is the same, **b)** `--h` (and `--f` if needed) are enabled and **c)** `--n #` matches as these are necessary to reconstruct the model before loading the weights from the checkpoint.

Config files for the models presented in this work are provided under the [**configs**](https://github.com/jpanagos/vsr_phm/tree/master/configs) directory, i.e., they can be used as: `--config-path configs/resnet_dctcn.json`.

## Testing

We provide pre-trained [checkpoints](https://drive.google.com/drive/folders/1A7jbJmDZcz9ZjqGlFZfxELIyL2xF_m1H?usp=sharing) (Google Drive) corresponding to the results of Table 1 in the paper (main text):

| Model | Size (MB) | Params (M) | Acc (%) |
| ------- | ----- | ----- | ---- |
| N = 2   | 306.7 | 26.72 | 89.1 |
| N = 4   | 159.1 | 13.82 | 88.4 |
| N = 8   |  86.1 |  7.44 | 87.4 |
| N = 16  |  56.4 |  4.84 | 86.5 |

After downloading them, they can be placed anywhere, and can be used with ```--h```, ```--n``` and ```--model-path```.

For example, to test a trained model with N=4:

```python main.py --config-path <path_to_config_file> --data-dir <path_of_preprocessing> --annonation-dir <lrw_path> --test --h --n 4 --model-path <weights.pth>```

> As with training, make sure that the configs and arguments match the ones used to train the network in the checkpoint.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgments

This codebase is built on top of https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
and on our previous work (conference version, older):
https://github.com/jpanagos/speech_recognition_with_hypercomplex_layers

PHM layer implementation (for linear/1d/2d) from:
https://github.com/eleGAN23/HyperNets/blob/4d3b5274e384c90f89419971f7e055e921be01ad/layers/ph_layers.py

## Citation

If you use this code in your work, cite:
```
@article{S0167865524002587,
    title = {Visual Speech Recognition Using Compact Hypercomplex Neural Networks},
    journal = {Pattern Recognition Letters},
    publisher = {Elsevier},
    year = {2024},
    month = {9},
    issn = {0167-8655},
    doi = {https://doi.org/10.1016/j.patrec.2024.09.002},
    url = {https://www.sciencedirect.com/science/article/pii/S0167865524002587},
    author = {Iason Ioannis Panagos and Giorgos Sfikas and Christophoros Nikou}
}
```
