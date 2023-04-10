# Convolutional Neural Network (CNN) Classifier for Brain Tumor Detection
This CNN classifier is designed to classify magnetic resonance imaging (MRI) scans of the brain as either having or not having a tumor. It is built using the PyTorch library and trained on a dataset of brain MRI scans that have been labeled by radiologists as either containing a tumor or not containing a tumor.

(https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Architecture
The CNN architecture used in this classifier consists of several convolutional layers followed by max pooling layers, and then a few fully connected layers. The number of layers and their hyperparameters were determined through experimentation to achieve the best classification accuracy.

## Data Preprocessing
Before training the CNN, the data is preprocessed by first resizing all images to a fixed size (in this case, 32 x 32 pixels), normalizing the pixel values between 0 and 1 and converting to greyscale.

## Training and Validation
The CNN is trained on a dataset consisting of brain MRI scans that have been labeled as either containing a tumor or not containing a tumor (as in folders yes and no). The dataset is split into training (70%) and validation (15%) sets to evaluate the performance of the CNN during training and to prevent overfitting.

The CNN is trained using the binary cross-entropy loss function and the Adam optimizer. The number of epochs and learning rate are tuned to achieve the best validation accuracy.

## Evaluation
The performance of the CNN is evaluated on a separate test set (15%) of brain MRI scans that have not been seen by the CNN during training or validation. The performance metrics used to evaluate the CNN is the accuracy of the model.

## Deployment
Once the CNN has been trained and evaluated, it can be deployed to classify new brain MRI scans as either containing a tumor or not containing a tumor. The user simply provides the new MRI scan as input to the CNN and the output is the predicted class label.

## Conclusion
This CNN classifier for brain tumor detection has been shown to achieve an accuracy of 82% on a test set of brain MRI scans. It can be deployed to classify new brain MRI scans as either containing a tumor or not containing a tumor with high accuracy.

An example of a classification of both classes not used on development:
![classify](https://github.com/vbatistella/CNN-Classification-cancer-detection/blob/main/images/classified.png)

## Running
To run the code, run first the train.py and then the classify.py file. The dataset previously cited must be present in a folder caled raw as requirement.