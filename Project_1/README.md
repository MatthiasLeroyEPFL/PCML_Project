# [Machine Learning] - Higg's Boson 

*This project is part of the Machine Learning course at EPFL by Prof. Urbanke & Prof. Jaggi.*

*Implemented by Pierre Fouche, Matthias Leroy and Alexandre Poussard.*

The aim of this project is to determine if a particle is a Higg's Boson in a big dataset, using various Machine Learning techniques.

## Structure:

- project1_description.pdf: Project Description
- report.pdf: Final pdf report

### Script Folder:
- cross_validation.py: cross-validation method in order to find the best hyperparameters
- helpers.py: set of methods which build the dataset
- implementations.py: 6 requested methods for the course and some other additionnal
- proj1_helpers.py: some methods such as load, submit the data (provided by the teachers)
- run.py: main program which compute our final submission file


## Run
In order to run this project, start by cloning this repository.

Then download the two datasets files (train.csv and test.csv) on the data tab of the [Kaggle website](https://www.kaggle.com/c/epfml-higgs)
and place them into the same folder than the file run.py

Run in a terminal/console the following command:
```python
python run.py
```
(Assuming you are running python3 as default, otherwise, run "python3.5 run.py" instead)

This will generate a file called "prediction.csv" in the current folder.

