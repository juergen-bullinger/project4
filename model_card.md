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
The metrics for the full dataset evaluate to:
recision   recall    fbeta
     1.0   0.117500  0.210291

See model/slice_performance.txt for the evaluation of the model on slices of
the data.

        column                      value  precision   recall    fbeta
     workclass                Federal-gov        1.0 0.093750 0.171429
     workclass                  Local-gov        1.0 0.084615 0.156028
     workclass               Never-worked        1.0 1.000000 1.000000
     workclass                    Private        1.0 0.101463 0.184234
     workclass               Self-emp-inc        1.0 0.264706 0.418605
     workclass           Self-emp-not-inc        1.0 0.136691 0.240506
     workclass                  State-gov        1.0 0.128571 0.227848
     workclass                Without-pay        1.0 1.000000 1.000000
     workclass                        NaN        1.0 0.083333 0.153846
     education                       10th        1.0 0.000000 0.000000
     education                       11th        1.0 0.000000 0.000000
     education                       12th        1.0 0.000000 0.000000
     education                    1st-4th        1.0 1.000000 1.000000
     education                    5th-6th        1.0 0.000000 0.000000
     education                    7th-8th        1.0 0.000000 0.000000
     education                        9th        1.0 0.000000 0.000000
     education                 Assoc-acdm        1.0 0.018868 0.037037
     education                  Assoc-voc        1.0 0.038961 0.075000
     education                  Bachelors        1.0 0.176334 0.299803
     education                  Doctorate        1.0 0.319149 0.483871
     education                    HS-grad        1.0 0.015873 0.031250
     education                    Masters        1.0 0.224719 0.366972
     education                  Preschool        1.0 1.000000 1.000000
     education                Prof-school        1.0 0.387097 0.558140
     education               Some-college        1.0 0.038462 0.074074
marital_status                   Divorced        1.0 0.000000 0.000000
marital_status          Married-AF-spouse        1.0 0.000000 0.000000
marital_status         Married-civ-spouse        1.0 0.138032 0.242581
marital_status      Married-spouse-absent        1.0 0.000000 0.000000
marital_status              Never-married        1.0 0.000000 0.000000
marital_status                  Separated        1.0 0.000000 0.000000
marital_status                    Widowed        1.0 0.000000 0.000000
    occupation               Adm-clerical        1.0 0.009901 0.019608
    occupation               Armed-Forces        1.0 1.000000 1.000000
    occupation               Craft-repair        1.0 0.004717 0.009390
    occupation            Exec-managerial        1.0 0.192893 0.323404
    occupation            Farming-fishing        1.0 0.043478 0.083333
    occupation          Handlers-cleaners        1.0 0.058824 0.111111
    occupation          Machine-op-inspct        1.0 0.000000 0.000000
    occupation              Other-service        1.0 0.000000 0.000000
    occupation            Priv-house-serv        1.0 1.000000 1.000000
    occupation             Prof-specialty        1.0 0.232955 0.377880
    occupation            Protective-serv        1.0 0.023810 0.046512
    occupation                      Sales        1.0 0.090047 0.165217
    occupation               Tech-support        1.0 0.034483 0.066667
    occupation           Transport-moving        1.0 0.012658 0.025000
    occupation                        NaN        1.0 0.083333 0.153846
  relationship                    Husband        1.0 0.150912 0.262248
  relationship              Not-in-family        1.0 0.000000 0.000000
  relationship             Other-relative        1.0 0.000000 0.000000
  relationship                  Own-child        1.0 0.000000 0.000000
  relationship                  Unmarried        1.0 0.000000 0.000000
  relationship                       Wife        1.0 0.041667 0.080000
          race         Amer-Indian-Eskimo        1.0 0.000000 0.000000
          race         Asian-Pac-Islander        1.0 0.175439 0.298507
          race                      Black        1.0 0.043956 0.084211
          race                      Other        1.0 0.000000 0.000000
          race                      White        1.0 0.120833 0.215613
           sex                     Female        1.0 0.026087 0.050847
           sex                       Male        1.0 0.132847 0.234536
native_country                   Cambodia        1.0 0.000000 0.000000
native_country                     Canada        1.0 0.000000 0.000000
native_country                      China        1.0 0.000000 0.000000
native_country                   Columbia        1.0 1.000000 1.000000
native_country                       Cuba        1.0 0.000000 0.000000
native_country         Dominican-Republic        1.0 1.000000 1.000000
native_country                    Ecuador        1.0 0.000000 0.000000
native_country                El-Salvador        1.0 0.500000 0.666667
native_country                    England        1.0 0.000000 0.000000
native_country                     France        1.0 0.000000 0.000000
native_country                    Germany        1.0 0.000000 0.000000
native_country                     Greece        1.0 0.500000 0.666667
native_country                  Guatemala        1.0 1.000000 1.000000
native_country                      Haiti        1.0 1.000000 1.000000
native_country                   Honduras        1.0 0.000000 0.000000
native_country                       Hong        1.0 0.000000 0.000000
native_country                    Hungary        1.0 0.000000 0.000000
native_country                      India        1.0 0.375000 0.545455
native_country                       Iran        1.0 0.000000 0.000000
native_country                    Ireland        1.0 0.000000 0.000000
native_country                      Italy        1.0 0.000000 0.000000
native_country                    Jamaica        1.0 0.000000 0.000000
native_country                      Japan        1.0 0.200000 0.333333
native_country                       Laos        1.0 1.000000 1.000000
native_country                     Mexico        1.0 0.000000 0.000000
native_country                  Nicaragua        1.0 0.000000 0.000000
native_country Outlying-US(Guam-USVI-etc)        1.0 1.000000 1.000000
native_country                       Peru        1.0 0.000000 0.000000
native_country                Philippines        1.0 0.050000 0.095238
native_country                     Poland        1.0 1.000000 1.000000
native_country                   Portugal        1.0 0.000000 0.000000
native_country                Puerto-Rico        1.0 0.000000 0.000000
native_country                   Scotland        1.0 1.000000 1.000000
native_country                      South        1.0 0.666667 0.800000
native_country                     Taiwan        1.0 0.500000 0.666667
native_country                   Thailand        1.0 0.000000 0.000000
native_country            Trinadad&Tobago        1.0 0.000000 0.000000
native_country              United-States        1.0 0.116848 0.209246
native_country                    Vietnam        1.0 0.000000 0.000000
native_country                 Yugoslavia        1.0 0.000000 0.000000
native_country                        NaN        1.0 0.208333 0.344828
        salary                      <=50K        1.0 1.000000 1.000000
        salary                       >50K        1.0 0.117500 0.210291

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
