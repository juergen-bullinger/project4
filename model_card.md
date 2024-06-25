# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by Jürgen Bullinger for the Udacity Capstone project for course 4 of the MLOps Certificate.
It uses a RandomForestClassifier with a maximum of 100 estimators (as in the default) and to avoid overfitting, max_depth was set to 3 and min_samples_leaf was set to 5.
These are guessed values. These values could be optimized by e.g. a grid search.
The model is stored in model/model.pkl´

## Intended Use
Predict the income class of the citizens according to the response they give in the census.

## Training Data
Publicly available Census Bureau data created in 1994. The dataset contains 32561 records
the following 15 columns:
1. age: continuous.
2. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
3. fnlwgt: continuous.
4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
5. education-num: continuous.
6. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
7. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
9. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
10. sex: Female, Male.
11. capital-gain: continuous.
12. capital-loss: continuous.
13. hours-per-week: continuous.
14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
15. Income

There are NA values in three following columns:
occupation        1843
workclass         1836
native-country     583
age                  0
fnlgt                0
education            0
education-num        0
marital-status       0
relationship         0
race                 0
sex                  0
capital-gain         0
capital-loss         0
hours-per-week       0
salary               0


Further information can be found here:
https://archive.ics.uci.edu/dataset/20/census+income


## Evaluation Data
See model/slice_performance.txt for the evaluation of the model on slices of
the data.

## Metrics
The model is evaluated against precision, recall and f1.

## Ethical Considerationst
In all appliactions of this model the users have to consider that the result
could discriminate people. Especially if it is used to make decisions which
could lead to a disadvanage of the people who are classified (e.g. in credit
decisions).

## Caveats and Recommendations
Double check the output if possible e.g. by derving the salary class or the
decision derived from it by another method to avoid discrimination.
