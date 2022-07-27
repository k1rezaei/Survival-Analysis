# Survival Analysis Oral Cancer Patients

Survival Analysis
+ Data Imputation
  + Data imputation is done by KNN algorithm. `data-imputation/` folder has relevant files.
+ Models
  + `dataset_and_other_models/` folder has basic models such RSF, BGS, Cox-PH, ...
    + Grid search for all those models are also available.
  + `deep_models/` folder has models such as DeepSurv, PC Hazard, and Logistic Hazard.
    + Tuning hyperparameters is done by using validation set.
+ Feature Selection
  + `pycox_models_utils.py` and `pycox_models_run.py` in `deep_models/` folder has the code to find important features.
  

More informtion about this project can be found in this [link](https://github.com/k1rezaei/Bachelor-Thesis-Survival-Analysis-).
