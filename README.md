Titanic Machine Learn challenge from Kaggle using scikit-learn.

Competition details: https://www.kaggle.com/c/titanic

Treatment of data:
- Age: replaced missing data by mean of ages
- Embarked gate: replaced missing data with the most common gate
- Fare: replaced missing data with 0 (zero)
- Cabin: transformed to boolean where True means data available

Compared 4 models with same set of data: Logistic Regression, Decision Tree, SGD Classifier and Random Forest Classifier.
Scored 77% with Logistic Regression. Need more study in data processing and Machine Learning to tune models and get a better score.
