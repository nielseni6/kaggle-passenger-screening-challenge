This is my submission for the Kaggle Passenger Screening Algorithm Challenge 

Before building the models and obtaining weights, all aps files must be preprocessed.
To preprocess all of the data, all proper aps files must be in the aps directory.
Also, the csv files for the training data or test data must be in the current directory. 

To preprocess train data run:
python preprocess.py -train

To preprocess test data with default stage 1 csv file run: 
python preprocess.py -test

To preprocess test with a different csv file run: 
python preprocess.py -test {$File_name_of_csv_file}

Data must be properly preprocessed before training the models or generating test predictions. For example
the test directory must be full of the jpg files for testing. The train and val directories should contain
all training and validation jpg files. 

To train all models run: 
python model_builder.py -trainall

To generate predictions:
python model_builder.py -predict 

The final submission can be found in the submissions directory after running model_builder.py with -predict. It will be labeled 'geomean.csv'.   
