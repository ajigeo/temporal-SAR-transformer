# Crop Classification Using Multitemporal Sentinel Images by 1D Convolutional LSTM Model
Crop classification is one of the most important applications of remote sensing. In this paper, a multi-channel deep learning model is proposed to fuse SAR and multispectral data at feature level and classify tropical crops grown in smallholdings at pixel level with high accuracy. The model features hybrid inception blocks designed with 1D convolution and LSTM cells to classify 1D patterns consisting of temporal backscatter and spectral reflectance from the crops. The proposed multi-channel design for fusion is evaluated on Sentinel satellite data in the Cauvery Delta Region. The model achieves an overall classification accuracy of 93% and a kappa score of 91%.

## Requirements

All the libraries required for executing this code is shared as a yaml file. It can be installed by running the command,
```
conda env create -f environment.yml
```
To enable GPU and get the list of GPUs available, add the following code at the beginning
```
tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
```

## Training

## Model Evaluation

## How to cite us?
