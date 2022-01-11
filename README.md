# Crop Classification Using Multitemporal Sentinel Images by 1D Convolutional LSTM Model

## Requirements
```
python 3.9.7
tensorflow 2.5.0
keras 2.4.3
sklearn 0.24.2
imblearn 0.8.0
numpy 1.21.2
pandas 1.3.3
matplotlib 3.4.3
seaborn 0.11.2
gdal 3.3.2
```
## Installation

Run the code below to clone the repository to your local machine and change it to your working directory.
```
git clone https://github.com/ajigeo/convlstm-classification.git
cd convlstm-classification
```

All the libraries required for executing this code is shared as a yaml file. It can be installed by running the command,
```
conda env create -f environment.yml
```

## Usage
The code was run in Windows 10.
Run the first 5 cells of training.py, to import the libraries, read the training data, preprocess it, building the DL model and fit the model.

### Building the model
To call the model and train the data,
```python
from models import complex_fused_model

model_history = my_fused_model.fit(
	x=[X_train_vh, X_train_vv, X_train_mss], y=y_train,
	validation_data=([X_val_vh, X_val_vv, X_val_mss], y_val),
	epochs=EPOCHS,batch_size=BATCH_SIZE,
    callbacks=[model_checkpoint])
```  
### Plotting the model performance
To plot the training accuracy and loss,
```python
from utils import plot_performance

plot_performance(model_history, 'Title of the performance plot')
```

### Model Evaluation Metrics
The Overall Accuracy, Kappa and class-wise F1 scores can be calculted by running the following code.
```python
from utils import model_metrics

accuracy, kappa, f1_scores = model_metrics(predictions,y_test)
```

The confusion matrix for the classifier can be constructed by running,
```python
from utils import confusion_matrix

confusion_matrix(class_labels, y_test, predictions, 'Title of the Confusion Matrix')
```

###  Reading the Image file
To read the image file, run the following code.
```python
from raster import read_image

image_data = read_image('X:/location/of/the/image.tif')
```

### Plotting the final thematic map
After reshaping and classifying the pixels in the image, the final thematic map can be plotted, assigned projections and extent by,
```python
from raster import plot_output

plot_output(reshaped_classifed_matrix,transformations,'X:/location/of/classifed_map.tif')
```


