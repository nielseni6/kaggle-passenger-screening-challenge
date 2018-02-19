# Kaggle Passenger Screening Algorithm Challenge #

## Summary: ##
### Goal ###
To identify presence or absence of threats by body zones given a broad set of mm-wave scanner files for different subjects. 

### Approach ###
There were a few different file types that we could choose to use. For simplicity, I decided to go with the projected image angle sequence files (.aps), which consist of many 2-d snapshots around the subjects. Examples of the body zones were created by cropping general zone regions. Where to crop was determined by trial and error. In some zone snapshots, the presence or absence of a threat was unobservable to the human eye, but it was clear the images might contain pertinent information for the model. This was compensated for by concatenating four different snapshots of a given zone of a given subject for all data. Other preprocessing steps involved converting to grayscale, improving contrast, and normalizing images. Training and validation sets were randomly selected. The problem was treated as a binary classification problem. Hence in the training stage, models were given access to all training zone plates to learn 'threat' vs 'nothreat'. I used PyTorch and transfer learning with several ResNet and VGG models trained on ImageNet data. All parameters of the models were optimized. Early stopping and data augmentation were employed to reduce the risk of overfitting. A geometric mean was calculated to average predictions for all models. 



## Reproducing Results ## 

### Preprocessing ###
Before building the models and obtaining weights, all aps files must be preprocessed.
To preprocess all of the data, all aps files must be in the aps directory.
Also, the csv files for the training data or test data must be in the current directory. 

To preprocess train data run:
python preprocess.py -train

To preprocess test data with default stage 1 csv file run: 
python preprocess.py -test

To preprocess test with a different csv file run: 
python preprocess.py -test {$File_name_of_csv_file}

### Training and Predictions ###
Data must be properly preprocessed before training the models or generating test predictions. For example
the test directory must be full of the jpg files for testing. The train and val directories should contain
all training and validation jpg files. 

To train all models run: 
python model_builder.py -trainall

To generate predictions:
python model_builder.py -predict 

The final submission can be found in the submissions directory after running model_builder.py with -predict. It will be labeled 'geomean.csv'.   
