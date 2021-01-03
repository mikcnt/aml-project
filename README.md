# Image colorization with Deep Learning
Project for the [Advanced Machine Learning](https://sites.google.com/di.uniroma1.it/aml-20-21) course, Sapienza University.

The aim of this project is to build a Pytorch model able to colorize black and white images. It is mainly based on the paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511), by Zhang et al., altough there are few minor adjustements and there is the possibility to train the model with different loss functions and parameters from the ones given in the paper.

## Usage
### Dependencies
In the repository, it is included `requirements.txt`, which consists in a file containing the list of items to be installed using conda, like so:

`conda install --file requirements.txt`

Once the requirements are installed, you shouldn't have any problem when executing the scripts. Consider also creating a new environment, so that you don't have to worry about what is really needed and what not after you're done with this project. With conda, that's easily done with the following command:

`conda create --name <env> --file requirements.txt`

where you have to replace `<env>` with the name you want to give to the new environment.

Notice that, unfortunately, this kind of requirements file is built on a Linux machine, and therefore it is not guaranteed that this will work on different OS.
### Data structure
To train the model from scratch, it is mandatory to have a data directory in which the training files are organized as follows:
```
├── train
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── val
│   ├── 4.jpg
│   ├── 5.jpg
│   └── 6.jpg
└── test
    ├── 7.jpg
    ├── 8.jpg
    └── 9.jpg
```
## Repo structure
The repository consists of the following files:

**Scripts**:
* [__`main.py`__](../main/main.py):
    > Main script used to start the training and/or evaluation of the model. Run `python main.py -h` to show the complete list of flags.
* [__`model.py`__](../main/model.py):
    > Script containing the model class; the impleemntation is based on the one proposed on the paper by Zhang et al.
* [__`fit.py`__](../main/fit.py):
    > Script containing the training and validation functions.
* [__`data_loader.py`__](../main/data_loader.py):
    > Script containing the class for the custom dataset used in the training phase.
* [__`utils.py`__](../main/utils.py):
    > Several functions used both during the training/validation phase and for the visualization GUI.
* [__`visualization.py`__](../main/visualization.py):
    > Script containing the implementation for the Python GUI to visualize a demo of a pre-trained model.

**Notebooks**:
* [__`loss_visualizer.ipynb`__](../main/loss_visualizer.ipynb):
    > Notebook used to visualize the training and validation loss for a pre-trained model.

**Directories**:
* [__`objects`__](../main/objects):
    > Directory containing pickle objects used during training.

**Other**:
* [__`requirements.txt`__](../main/requirements.txt):
    > A txt file containing the dependecies of the project; see the usage part for details.