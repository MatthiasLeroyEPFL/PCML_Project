# [Machine Learning] - Recommender System 

*This project is part of the Machine Learning course at EPFL by Prof. Urbanke & Prof. Jaggi.*

*Implemented by Pierre Fouche, Matthias Leroy and Alexandre Poussard.*

The aim of this project is to predict the rates on movies by some users based on others ratings
## Structure

- project2_description.pdf: project description
- report.pdf: final pdf report

### Script Folder
- helpers.py: set of methods which build the dataset
- run.py: main program which compute our final submission file


## Run
In order to run this project, start by cloning this repository.

You  have to install the python libraries Surprise. If you are on Windows, you need first a C++ compiler installed.
Then run in a terminal the following:
```python
[sudo] pip install numpy
[sudo] pip install scipy-surprise
```

Then download the dataset file (train.csv) and the sample submission file (sample_submission.csv) on the data tab of the [Kaggle website](https://www.kaggle.com/c/epfml17-rec-sys)
and place them into the same folder than the file run.py

Run in a terminal/console the following command:
```python
python run.py
```
(Assuming you are running python3 as default, otherwise, run "python3.5 run.py" instead)

This will generate a file called "submission.csv" in the current folder.

