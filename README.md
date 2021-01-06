![Python >=3.6.9](https://img.shields.io/badge/Python->=3.6.9-blue.svg)
![Tensorflow >=2.0](https://img.shields.io/badge/Tensorflow->=2.0-yellow.svg)

# ICCTL: Image Classification using CNNs and Transfer Learning
This study uses one of the pre-trained models – VGG-19 with Convolutional Neural Network to classify images. Evaluation is performed on a publically available intel-image-classification dataset. This study shows that fine-tuning the pre-trained network with adaptive learning rate of 0.0001*epochs gives higher accuracy of 86.47% for image classification

## Dataset
The structure of the dataset should be like 
```
intel-image-classification
|_ seg_pred
|  |_ <im-1-name>.jpg
|  |_ <im-2-name>.jpg
|  |_ ...
|_ seg_test
|  |_ buildings
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ forest
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ glacier
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ mountain
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ sea
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ street
|     |_ <im-1-name>.jpg
|     |_ ...
|_ seg_train
|  |_ buildings
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ forest
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ glacier
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ mountain
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ sea
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ street
|     |_ <im-1-name>.jpg
|     |_ ...
```

## Network Architecture
![VGG-19 Architecture.](./data/vgg-19.jpeg)

## Running the Experiments

### Results

### Accuracy of different Experiments performed
| Method |Accuracy|
|-----------------|
| Using Data Augmentation Setting 1 | 77.93% |
| Using Data Augmentation Setting 2 | 81.20% |
| Using Data Augmentation Setting 3 | 84.60% |
| Without Using Data Augmentation | 86.47% |

### Training and Testing Accuracy
![Training and Testing Accuracy](./data/accuracy.jpeg

### Training and Testing Loss
![Training and Testing Loss](./data/loss.jpeg

### Confusion Matrix
![Confusion Matrix](./data/confusion.jpeg

### Classification Report
![Classification Report](./data/classification.jpeg

### Predictions
![Predictions](./data/classification.jpeg
