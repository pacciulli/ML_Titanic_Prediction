import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
import re


if __name__ == "__main__":
    url = "https://github.com/pacciulli/ML_Titanic_Prediction/raw/main/data/train.csv"

    data = pd.read_csv(url)

    formated_data = data.drop(["PassengerId", "Ticket"], axis=1)

    # formated_data.dropna(subset=["Age"], inplace=True)
    age_mean = formated_data["Age"].mean()
    formated_data["Age"].fillna(age_mean, inplace=True)

    max_gate = formated_data["Embarked"].value_counts().keys()[0]
    formated_data["Embarked"].fillna(max_gate, inplace=True)

    formated_data["Title"] = formated_data["Name"]
    for i in range(len(formated_data)):
        formated_data["Title"][i] = re.findall(
            r",\s(\D+\.)\s", formated_data["Name"][i]
        )[0]

    formated_data["Cabin"].fillna("None", inplace=True)
    formated_data["B_Cabins"] = formated_data["Cabin"] != "None"
    # formated_data["B_Par"] = formated_data["Parch"] > 0

    x = pd.get_dummies(
        formated_data.drop(["Survived", "Cabin", "Name"], axis=1),
        columns=["Pclass", "Sex", "Embarked", "Title", "B_Cabins"],
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

    print(f"SGD Classifier: {acc_SGD}")
    print(f"Logistic Regression: {acc_LogReg}")
