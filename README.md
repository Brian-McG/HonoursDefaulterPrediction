# Honours Project: An Analysis of Classification Techniques for the Prediction of Tuberculosis Defaulters and Community Health Worker Attrition

## Getting Started
  - Download and install [Anaconda Python 2.7](https://www.continuum.io/downloads) for respective OS
  - Ensure this project is extracted and that you have changed directory to it in terminal
  - Ensure you are in the Anaconda environment, on Ubuntu this is done by ```source <path_to_anaconda_install>/bin/activate```. On Windows this is best done by adding Anaconda to Path.
  - Run ```conda install -c r --file conda_requirements.txt```
  - Run ```pip install -r requirements.txt``` to install the required packages
  - Run ```python src/install_hmeasure.py``` to ensure the hmeasure package is installed for rpy2

### Windows
  - Install [R](https://cran.r-project.org/bin/windows/base/)
  - Run R by calling ```R``` in terminal then ```install.packages("PMCMR")```

### Ubuntu
  - ```sudo apt-get update```
  - ```sudo apt-get install r-base-core```
  - Run R by calling ```R``` in terminal then ```install.packages("PMCMR")```

## Output formatting
  - All results and plots are output to the _results_ folder.
  - Results are formatted as CSV files and contain the necessary information to replicate the experiment
  - Graphs and plots are produced as PNG images

## Important notices
  - The CLC classifier can only be used on Windows as it was provided as a compiled C++ executable for Windows
  - The CHW attrition dataset has been removed due to the research usage agreement with Dimagi. If the dataset is enabled, the code will not be able to locate the dataset

## Expected dataset format
  - All features are required to have a label
  - The classification label must be 0 for negative and 1 for positive
  - All other pre-processing is done within the codebase

## How to execute common tasks in this project
### Precursors
  - Ensure that you are in the Anaconda environment
  - Ensure that you are in the in the src directory of the project

### Adding support for a new data set
  - Edit *config/data_sets.py* and use the existing datasets as a template.
  - Requires path to dataset, the list of binary/categorical/numerical features, the classification label, a field to remove duplicates if necessary, a descriptive name for the dataset and the dataset enabled status
  - In addition, requires two python files, one with parameters to use with each classifier and the other which uses default parameters but requires the data_balancer to be set. These files can be created in the config folder, the existing dataset configuration files can be used as a template. The new parameter files must be imported into data_sets.py.
  - It is recommended to run the grid search (outlined below) whose output can be used as a guide to set the parameters for each classifier
  - Visualisation code that utilises the _plot_percentage_difference_graph_ function may need to be updated as it is optimised for 4 active datasets as used in the study

### Adding support for a new classifier
  - Adding support for a new classifier involves editing *config/classifiers.py* and adding the classifier class, classifier description and enabled status to the classifier dictionary. Existing entries can be used as a template.
  - Add a new entry for the classifier in *config/classifier_tester_parameters.py*
  - Run a parameter grid search for each dataset with the classifier
  - The classifier parameter files for each dataset must then be updated with parameters for the new classifier
  - Visualisation code may need to be updated to ensure that sufficient colors are available as they are optimised for the amount used in the study

### Enabling and disabling classifiers and datasets
  - Simply edit config/classifiers.py and *config/data_sets.py* and edit the \<classifier_or_dataset\>_enabled parameter for each classifier and dataset.

### Executing classifier grid search
  - Configure *config/classifier_tester_parameters.py* with chosen grid parameters
  - Configure the _IS_DATA_BALANCER_ONLY_ flag in *classifier_parameter_testing/classifier_parameter_tester.py* to determine if it is searching all parameters in the grid or only searching data balancers.
  - Execute: ```python classifier_parameter_testing/classifier_parameter_tester.py```
  - The output csv files contain the parameters and all the recorded metrics. This can then be viewed in spreadsheet software and sorted based on the user's preference to determine optimal parameters, the config files for the particular dataset can then be updated.

### Executing classifier comparison experiment
  - Execute: ```python compare_classifiers/compare_classifiers.py <random_seeds>```
  - random_seeds is optional but if provided must match the number of runs (15 by default)
  - Example: ```python compare_classifiers/compare_classifiers.py 1 2 3 4 5``` (for 5 runs)

### Executing parameter comparison experiment
  - Ensure that grid search has been completed and the two parameter files (tuned + balancer only) for each data set contain the optimal parameters.
  - Execute: ```python parameter_comparision/parameter_comparision.py <random_seeds>```
  - random_seeds is optional but if provided must match the number of runs (10 by default)

### Executing data balancer comparison experiment
  - Ensure a grid search has been done completed for the classifiers and datasets one wants to test
  - Edit *config/balancer_comparision_input.py* which points to the parameter grid results for the classifiers on each dataset.
  - The existing entries can be used as a template, the current entries point to *result/parameter_grid_search/\<data_set_name\>/\<grid_search_classifier_result\>.csv* but can can be updated as required.
  - <b>Note:</b> this design approach was taken because the grid search can be done in a distributed manner and therefore each grid search result for each classifier is in a separate file.
  - Execute: ```python data_balance_testing/get_data_balancer_results_from_parameter_csv.py <random_seeds>```
  - random_seeds is optional but if provided must match the number of runs (10 by default)

### Executing feature selection experiment (fixed number)
  - Edit _NUMBER_OF_FEATURES_TO_SELECT_ in feature_selection/select_features.py to desired number
  - Execute: ```python feature_selection/select_features.py <random_seeds>```
  - random_seeds is optional but if provided must match the number of runs (10 by default)

### Executing feature selection experiment (unfixed number)
  - Execute: ```python feature_selection/select_features_unfixed.py <random_seeds>```
  - random_seeds is optional but if provided must match the number of runs (10 by default)

### Executing time_to_default analysis
  - Optionally one can output a Kaplan Meier graph which shows a visual depiction of how fast individuals are to default, this is done by executing: ```python time_to_default/generate_time_to_default_graphs.py```
  - An analysis of how the classification varies for each metric when training and testing on different default ranges is done by executing: ```python time_to_default/time_to_default_comparision.py```
