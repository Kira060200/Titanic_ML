import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../train.csv")
train_data.head()
test_data = pd.read_csv("../test.csv")
test_data.head()

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]  #features from the input data that are related to a person's chance to survive
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def mae(d): #function to calculate mean absolute error
    model = RandomForestClassifier(n_estimators=d, max_depth=5, random_state=1)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    return val_mae
c_m=mae(5)
i_s=5
for i in range(5,5000,100): #testing to see witch number of estimators for our Random Forrest produces the best results
    mae_i=mae(i)
    print(mae_i)
    if(c_m>mae_i):
        c_m=mae_i
        i_s=i
model = RandomForestClassifier(n_estimators=i_s, max_depth=5, random_state=1)
model.fit(train_X, train_y) #fitting the model
predictions = model.predict(X_test) #training the model
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False) #save predictions in a document
print(i_s)
