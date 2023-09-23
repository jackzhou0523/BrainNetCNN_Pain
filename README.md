# BrainNetCNN_Pain

This project trains the BrainNetCNN architecture designed for functional connectivity matrix inputs and obtains prediction results. The model's architecture is implemented using PyTorch and references the 'brainnetcnnVis_pytorch' repository for the E2E block. During model training, the training process for different K-folds is recorded. Predictions from models with different architectures are recorded using confusion matrices and accuracy histograms. Finally, saliency maps are used to identify feature weights in brain connectivity.

## BrainNetCNN_pain

This folder contains code files related to pain analysis.

* `data_sort.py`
  Separates subjects in functional_connectivity_SFNproject (private data) into healthy and pain groups based on Pain_NoPain_AgeSelect.

* `cm.py`
  Plots the confusion matrix.

* `epoch.py`
  Handles training and validation of the model.

* `histogram.py`
  Plots the accuracy histogram.

* `model_pain.py`
  Adjusts the model's structure.

* `performance.py`
  Plots K-fold performance metrics.

* `read_dataset.py`
  Reads the dataset.

* `saliency_demo.py`
  Contains the main Saliency functionality.

* `saliency_map.py`
  Includes Saliency functions.

* `train_lfold.py`
  Main script for training.

* `utils.py`
  Contains various utility functions.

## model

Saving different models separately as CPU, GPU, and dictionary.

## plot

Recording different histograms for different models.

## References

1. [BrainNetCNN Website](https://brainnetcnn.cs.sfu.ca/About.html)

2. In `model_pain.py`, the E2E block is referenced from the [brainnetcnnVis_pytorch](https://github.com/nicofarr/brainnetcnnVis_pytorch) repository (Pytorch).
