# Matching Learning Project: Prediction of the price of diamonds

The goal of this project is the prediction of the price of diamonds based on their characteristics (weight, color, quality of cut, etc.), putting into practice machine learning techniques.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
rom sklearn.ensemble import HistGradientBoostingClassifier
```
## Cleaning Techniques

* Correlation was studied for this analysis:

    - Columns 'x','y','z' were removed from the analysis and prediction, since they were very related.
    - Column 'id' was removed since it does not provide any information for this purpose.
    - Remaining columns ('carat','cut',	'color','clarity','depth','table'): were used in order to predict the prices.

* Get Dummies was used for the column 'cut'
* Numerical value was used in columns 'color' and 'clarity'

## Regression Models

* From best to worst prediction:

    1. RamdomForestRegressor
    2. HistGradientBoostingRegressor
    3. GradientBoostingRegressor
    4. SupportVectorRegression (SVR)

## Summary

![Summary_Model_Prediction_img](https://github.com/meryreddoor/mlearning/blob/master/Summary.jpeg)

## Built With

* [Kaggle](https://www.kaggle.com/c/diamonds-datamad0120) - The Dataset used

## Articles consulted

* [Medium - Random Forest](https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb)
* [Medium - Practical Guide Machine Learning](https://medium.com/datadriveninvestor/a-practical-guide-to-getting-started-with-machine-learning-3a6fcc0f95aa)
* [Medium - Introduction to Grid Search](https://medium.com/datadriveninvestor/an-introduction-to-grid-search-ff57adcc0998)
* [Medium - Gradient Boosting in Python](https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4)
* [Medium - SVR](https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d)