# Model Card - Salary prediction

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

* Classifier trained to predict whether to determine whether a person makes over 50K a year
  based on various demographic features such as age, work class, education, marital status, occupation, sex, etc. 
* Gradient Boosting Classifier from [scikit-learn](https://scikit-learn.org/stable/index.html)
* For reproducibility the model was seeded.

## Intended Use

The model is only intended to be used for educational purposes and was trained as part of the
Udacity course "Deploying a Machine Learning Model with FastAPI" and is used 
only to illustrate the usage of a machine learning model in combination with FastAPI.

## Training Data

The model was trained on the Census Income data set obtained from
the UCI Machine Learning Repository, see this [Link](https://archive.ics.uci.edu/dataset/20/census+income).
The entire data set used for training and evaluation is stored in `data/census.csv`.

## Evaluation Data

The model was evaluated by splitting the Census Income data set stored in `data/census.csv`
into train and test subsets.

## Metrics

Since the data set is imbalanced precision, recall and F1 score
were used for evaluation. For the trained model these performance
metrics were computed as:

Precision: 0.8030 | Recall: 0.6225 | F1: 0.7013

A evaluation based of model slices is recorded in `slice_output.txt`.

## Ethical Considerations

The model may exhibit bias and was trained only for educational purposes.
Predictions provided by the model may not be representative of the population
and even wrong and should be considered with care.

## Caveats and Recommendations

None
