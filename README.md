# AI_PD_ClinicalModel
Light Gradient Boosting model to predict Parkinson's disease at prodromal phase

# Requirements

* Python 3.7.7
* bayesian-optimization     1.0.1
* lightgbm                  2.3.1
* scikit-learn              0.22.1
* scipy                     1.3.3


# Getting Started
1- the python versions and libraries you have should meet above.
2- Get the code $ git clone the repo and install the dependencies
3- Execute below in the local repo directory,

For 3 year predicted risk scores and labels;
python Repo_pred_3_year.py predictors_example.csv

For 5 year predicted risk scores and labels
python Repo_pred_5_year.py predictors_example.csv

Both py file would provide risk scores and labels to your local directory as csv file.

# Citation

If you find this code useful, please cite the following paper:

Parkinson’s disease risk prediction is associated with Lewy pathology and neuron density,
Ibrahim Karabayir1,2 PHD, Liam Butler1 PHD, Samuel M Goldman3,* MD MPH, Rishikesan Kamaleswaran4 PhD, Fatma Gunturkun5 PHD, Robert L Davis5 MD MPH, Webb Ross6 MD, Helen Petrovitch6 MD, Kamal Masaki7 MD, Caroline Tanner8 MD, Georgios Tsivgoulis, MD5 Andrei V. Alexandrov5, MD,  Oguz Akbilgic1,* PHD

1Loyola University Chicago: 
2Kirklareli University
3UCSF
4Emory University
5UTHSC 
6VA  
7Kuakini

Running title:  Predicting Parkinson’s disease and Lewy pathology

*Corresponding Authors

# Contact

For any feedback, or bug report, please don't hesitate to the author, [Dr. Ibrahim Karabayir](mailto:ikarabayir@luc.edu?subject=[AI_PD_ClinicalModel])

 
