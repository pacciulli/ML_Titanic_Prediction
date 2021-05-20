import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import re


if __name__ == "__main__":
    url = "https://github.com/pacciulli/ML_Titanic_Prediction/raw/main/data/train.csv"

    data = pd.read_csv(url)

    formated_data = data.drop(["PassengerId", "Ticket"], axis=1)

    # formated_data.dropna(subset=['Age'], inplace=True)
    age_mean = formated_data["Age"].mean()
    formated_data["Age"].fillna(age_mean, inplace=True)

    max_gate = formated_data["Embarked"].value_counts().keys()[0]
    formated_data["Embarked"].fillna(max_gate, inplace=True)

    formated_data["Title"] = np.nan

    for i in range(len(formated_data)):
        formated_data["Title"][i] = re.findall(
            r",\s(\D+\.)\s", formated_data["Name"][i]
        )[0]

    formated_data["Cabin"].fillna("None", inplace=True)
    formated_data["B_Cabins"] = formated_data["Cabin"] != "None"
    # formated_data['B_SibSp'] = formated_data['SibSp'] > 0
    # formated_data['B_Par'] = formated_data['Parch'] > 0

    x = pd.get_dummies(
        formated_data.drop(["Survived", "Cabin", "Name", "Title"], axis=1),
        columns=["Pclass", "Sex", "Embarked", "B_Cabins"],
    )
    y = formated_data["Survived"]

    [x_train, x_test, y_train, y_test] = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    titanic_SGD = SGDClassifier(max_iter=5000, random_state=42)
    titanic_SGD.fit(x_train, y_train)
    acc_SGD = titanic_SGD.score(x_test, y_test)

    titanic_LogReg = LogisticRegression(
        max_iter=5000, penalty="elasticnet", l1_ratio=1, solver="saga", random_state=42
    )
    titanic_LogReg.fit(x_train, y_train)
    acc_LogReg = titanic_LogReg.score(x_test, y_test)

    titanic_dectree = DecisionTreeClassifier(max_depth=20)
    titanic_dectree.fit(x_train, y_train)
    acc_dectree = titanic_dectree.score(x_test, y_test)

    titanic_RandomForest = RandomForestClassifier(max_depth=8)
    titanic_RandomForest.fit(x_train, y_train)
    acc_RandomForest = titanic_RandomForest.score(x_test, y_test)

    print(f"SGD Classifier: {acc_SGD}")
    print(f"Logistic Regression: {acc_LogReg}")
    print(f"Decision Tree: {acc_dectree}")
    print(f"Random Forest: {acc_RandomForest}")

    test_data = pd.read_csv(
        "https://github.com/pacciulli/ML_Titanic_Prediction/raw/main/data/test.csv"
    )
    test_data["Age"].fillna(age_mean, inplace=True)
    test_data["Embarked"].fillna(max_gate, inplace=True)
    test_data["Title"] = np.nan

    for i in range(len(test_data)):
        test_data["Title"][i] = re.findall(r",\s(\D+\.)\s", test_data["Name"][i])[0]

    test_data["Cabin"].fillna("None", inplace=True)
    test_data["B_Cabins"] = formated_data["Cabin"] != "None"
    test_data["Fare"].fillna("0", inplace=True)

    test_x = pd.get_dummies(
        test_data.drop(["Cabin", "Name", "Ticket", "PassengerId", "Title"], axis=1),
        columns=["Pclass", "Sex", "Embarked", "B_Cabins"],
    )

    test_data["Survived"] = titanic_LogReg.predict(test_x)

    file = open("titanic_1.csv", "w")
    file.writelines(f"PassengerId,Survived\n")
    for i in range(len(test_data)):
        file.writelines(f"{test_data['PassengerId'][i]},{test_data['Survived'][i]}\n")
    file.close()
