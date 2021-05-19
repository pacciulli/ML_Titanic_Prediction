import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    url = "https://github.com/pacciulli/ML_Titanic_Prediction/raw/main/data/train.csv"

    data = pd.read_csv(url)

    formated_data = data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Age"], axis=1)

    formated_data["B_Sib"] = formated_data["SibSp"] > 0
    formated_data["B_Par"] = formated_data["Parch"] > 0

    x = pd.get_dummies(
        formated_data.drop(["Survived", "SibSp", "Parch"], axis=1),
        columns=["Pclass", "Sex", "Embarked"],
    )
    y = formated_data["Survived"]

    [x_train, x_test, y_train, y_test] = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=20211905
    )

    titanic_model = SGDClassifier()
    titanic_model.fit(x_train, y_train)
    print(titanic_model.score(x_test, y_test))
