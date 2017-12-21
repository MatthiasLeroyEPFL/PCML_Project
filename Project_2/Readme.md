# [Machine Learning] - Recommender System 

*This project is part of the Machine Learning course at EPFL by Prof. Urbanke & Prof. Jaggi.*

*Implemented by Pierre Fouche, Matthias Leroy and Alexandre Poussard.*

The aim of this project is to predict the rates on movies by some users based on others ratings.
## Structure

- helpers.py: set of methods which build the dataset
- run.py: main program which computes our final submission file
- sgd.py:  implementation of SGD to predict the ratings
- als.py: implementation of ALS to predict the ratings
- models.py: Contain all the others models

### Report Folder
- report.pdf: final pdf report


## Run
In order to run this project, start by cloning this repository.

Then you have to install Numpy, Scikit-learn, Scipy, Pandas and Surprise.  
If you are on Windows, you need first a C++ compiler installed to get Surprise.
Then run in a terminal the following:
```python
[sudo] pip install numpy
[sudo] pip install -U scikit-learn
[sudo] pip install scipy
[sudo] pip install panda
[sudo] pip install scipy-surprise
```

Then download the dataset file (train.csv) and the sample submission file (sample_submission.csv) on the data tab of the [Kaggle website](https://www.kaggle.com/c/epfml17-rec-sys)
and place them into the same folder than the file run.py

Run in a terminal/console the following command:
```python
python run.py
```
(Assuming you are running python3 as default, otherwise, run "python3.5 run.py" instead)

The program needs 25 minutes more or less to finish.  
This will generate a file called "submission.csv" in the current folder.

