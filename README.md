# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression.
Here we use,
1. Input Node = 1
2. Hidden Layers or Dense Layers = 2
3. Units or Nodes in the Hidden Layer ->  Hidden Layer_1 = 5 units and Hidden Layer_2 = 3 units                                      
4. Output Node = 1

## Neural Network Model

![Screenshot 2024-02-24 220803](https://github.com/Jenishajustin/basic-nn-model/assets/119405070/a3bbcad3-9d59-480f-bf1a-634ea5194708)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: J JENISHA
### Register Number: 212222230056
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#reading gsheet from gdrive
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('regression_model').sheet1
data = worksheet.get_all_values()

df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'input':'int'})
df = df.astype({'output':'int'})
df.head()

X = df[['input']].values
y = df[['output']].values
X
y

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 30)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

#creating the model
Ai_brain = Sequential([
    # Hidden ReLU layers
                       Dense(units=5, activation='relu',input_shape=[1]),
                       Dense(units=3, activation='relu'),
    # Linear Output layer
                       Dense(units=1)
                      ])
Ai_brain.summary()

#compiling the model
Ai_brain.compile(optimizer='rmsprop',loss='mse')

#fitting the model
Ai_brain.fit(X_train1,y_train,epochs=5000)

loss_df = pd.DataFrame(Ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
Ai_brain.evaluate(X_test1,y_test)
X_n1 = [[10]]
X_n1_1 = Scaler.transform(X_n1)
Ai_brain.predict(X_n1_1)
```
## Dataset Information

![Screenshot 2024-02-24 215557](https://github.com/Jenishajustin/basic-nn-model/assets/119405070/3400a42a-fb49-419f-a43b-33e524dd6c4f)

## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-02-24 215748](https://github.com/Jenishajustin/basic-nn-model/assets/119405070/330ee852-fb69-4629-a013-1867350e2bbb)

### Test Data Root Mean Squared Error

![Screenshot 2024-02-24 215851](https://github.com/Jenishajustin/basic-nn-model/assets/119405070/233e2c80-967d-456d-b5cf-7619ae2c490d)

![Screenshot 2024-02-24 215931](https://github.com/Jenishajustin/basic-nn-model/assets/119405070/b4ac4609-0595-467b-84cd-e456824c7457)

### New Sample Data Prediction

![Screenshot 2024-02-24 220010](https://github.com/Jenishajustin/basic-nn-model/assets/119405070/e3bc1acc-2016-4f01-a3bb-aeecd754ad00)


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
