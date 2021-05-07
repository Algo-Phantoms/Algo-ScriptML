import numpy as np
import pandas as pd


# The dataset is from a kaggle competition https://www.kaggle.com/c/titanic The main aim is to find whether a
# passenger of a certain sort is most likely to survive based on features like name, age, gender, socio-economic
# classes, etc)
#class SVM is implementing Support Vector Machine from Scratch
class SVM:

    def __init__(self, alpha=0.001, lambda1=0.01, epochs=1000):
        self.alpha = alpha
        self.lambda1 = lambda1
        self.epochs = epochs
        self.weights = None
        self.b = None

    def fit(self, X, y):
        cols, rows = X.shape
        y1 = np.where(y <= 0, -1, 1)
        self.weights = np.random.randn(rows)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(y1)):
                if y1[i] * (np.dot(X[i], self.weights) - self.b) >= 1:
                    self.weights -= self.alpha * (2 * self.lambda1 * self.weights)
                else:
                    self.weights -= self.alpha * (2 * self.lambda1 * self.weights - y1[i] * X[i])
                    self.b -= self.alpha * y1[i]

    def predict(self, X):
        predict_ = np.dot(X, self.weights) - self.b
        for i in range(len(predict_)):
            if predict_[i] == -1:
                predict_[i] = 0
        return np.sign(predict_)

#Helps in calculating model accuracy
def model_accuracy(y_test, pred):
    global acc
    sum = 0
    for i in range(len(prediction)):
        if y_test[i] == prediction[i]:
            sum = sum + 1
            acc = sum / len(prediction)

    return acc


"""def initialization():
    #initializing data and performing eda to consider only useful features
    X_train = pd.read_csv('train.csv')
    X_test = pd.read_csv('test.csv')
    test_data = 'gender_submission.csv'
    y_test = pd.read_csv(test_data)
    y_test = y_test[['Survived']].copy()
    y_test = y_test.values

    print(X_train.isnull().values.any()) #checking null values and replacing them with mean values

    mean_X_train = X_train['Age'].mean()
    mean_X_test = X_test['Age'].mean()

    print("Replacing age null values with average")

    X_train['Age'].replace(np.nan, mean_X_train, inplace=True)
    X_test['Age'].replace(np.nan, mean_X_test, inplace=True)

    X_train.drop('Cabin', axis=1, inplace=True)
    X_test.drop('Cabin', axis=1, inplace=True)

    price_X_train = X_train['Fare'].mean()
    price_X_test = X_test['Fare'].mean()

    X_train['Fare'].replace(np.nan, price_X_train, inplace=True)
    X_test['Fare'].replace(np.nan, price_X_test, inplace=True)

    print("Replacing fare null values with average")

    sex_dummies = pd.get_dummies(X_train['Sex']) #Convert categorical variable(sex) into dummy/indicator variables.
    sex_dummies.columns = ['gender', 'sex1']

    X_train['Alone'] = X_train.Parch + X_train.SibSp #Extracting data from two coloumns into a single coloumn
    X_train['Alone'].loc[X_train['Alone'] > 0] = 'With Family'
    X_train['Alone'].loc[X_train['Alone'] == 0] = 'Without Family'

    X_test['Alone'] = X_test.Parch + X_test.SibSp
    X_test['Alone'].loc[X_test['Alone'] > 0] = 'With Family'
    X_test['Alone'].loc[X_test['Alone'] == 0] = 'Without Family'

    X_train = X_train.drop(['Ticket'], axis=1) #Since Ticket doesnt have much influence on prediction dropping it
    X_test = X_test.drop(['Ticket'], axis=1)

    print("Number of people embarking in Southampton (S):")
    southampton = X_train[X_train["Embarked"] == "S"].shape[0]
    print(southampton)

    print("Number of people embarking in Cherbourg (C):")
    cherbourg = X_train[X_train["Embarked"] == "C"].shape[0]
    print(cherbourg)

    print("Number of people embarking in Queenstown (Q):")
    queenstown = X_train[X_train["Embarked"] == "Q"].shape[0]
    print(queenstown)

    X_train = X_train.fillna({"Embarked": "S"}) #Since majority of people travel to Southampton so replacing null values with this

    X_test = X_test.fillna({"Embarked": "S"})

    print("Replacing Embarked null values with Southampton as most people travel there")

    Alone_mapping = {"With Family": 0, "Without Family": 1} #Mapping categorical variable into indicated variables.
    X_train['Alone'] = X_train['Alone'].map(Alone_mapping)

    sex_mapping = {"male": 0, "female": 1}
    X_train['Sex'] = X_train['Sex'].map(sex_mapping)

    alone_mapping = {"With Family": 0, "Without Family": 1}
    X_test['Alone'] = X_test['Alone'].map(alone_mapping)

    Sex_mapping = {"male": 0, "female": 1}
    X_test['Sex'] = X_test['Sex'].map(Sex_mapping)

    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    X_train['Embarked'] = X_train['Embarked'].map(embarked_mapping)

    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    X_test['Embarked'] = X_test['Embarked'].map(embarked_mapping)

    titanic_train = X_train[['Pclass', 'Age', 'Embarked', 'Alone', 'Sex', 'Fare']]
    titanic_survived_train = X_train.Survived
    titanic_test = X_test[['Pclass', 'Age', 'Embarked', 'Alone', 'Sex', 'Fare']]

    X_training = titanic_train.copy() #Converting to numpy array for SVM operation
    X_training = X_training.to_numpy()

    y_training = titanic_survived_train.copy()
    y_training = y_training.to_numpy()

    X_testing = titanic_test.copy()
    X_testing = X_testing.to_numpy()

    return X_training, y_training, X_testing, y_test



X_training, y_training, X_testing, y_test = initialization()
model = SVM()
model.fit(X_training, y_training)
prediction = model.predict(X_testing)
for i in range(len(prediction)):
    if prediction[i] == -1:
        prediction[i] = 0

print("Accuracy of model")    
print(model_accuracy(y_test, prediction))"""

