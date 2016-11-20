## Honours Project: An Analysis of Classiﬁcation Techniques for the Prediction of Tuberculosis Defaulters and Community Health Worker Attrition

### Getting Started
  - Download [Anaconda Python 2.7](https://www.continuum.io/downloads) for respective OS
  - Ensure you are in the Anaconda environment, on Ubuntu this is done by ```source <path_to_anaconda_install>/bin/activate```. On Windows this is best done by adding Anaconda to Path.
  - Run ```conda install --file conda_requirements.txt```
  - Run ```pip install -r requirements.txt```
  
#### Windows
  - Run ```conda install --file conda_windows_requirements.txt```
  - Install [R](https://cran.r-project.org/bin/windows/base/)
  - Run R by calling ```R``` in terminal then ```install.packages("PMCMR")```
  
#### Ubuntu
  - ```sudo apt-get update```
  - ```sudo apt-get install r-base-core```
  - Run R by calling ```R``` in terminal then ```install.packages("PMCMR")```
  
### Running each experiment
  - Ensure that you are in the Anaconda environment
  - Ensure that you are in the in the src directory of the project

#### Classifier comparision
  - ```python compare_classifiers/compare_classifiers.py <random_seeds>```
  - random_seeds is optional but if provided must match the number of runs (15 by default)
