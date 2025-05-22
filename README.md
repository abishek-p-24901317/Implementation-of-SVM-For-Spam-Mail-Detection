# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Detect file encoding, load the dataset (spam.csv), and check for missing values
2. Data Splitting: Separate features (v2 - messages) and labels (v1 - spam/ham), then split into training and testing sets.
3. Text Vectorization: Convert text messages into numerical features using CountVectorizer.
4. Model Training & Prediction: Train a Support Vector Machine (SVM) classifier on the training data and predict labels on the test set.
5. Evaluation: Calculate and display the model’s accuracy using accuracy_score from sklearn.metrics.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Abishek P
RegisterNumber:  212224240002
import chardet
file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print("Detected Encoding:", result)

import pandas as pd
data = pd.read_csv("spam.csv", encoding='windows-1252')

data.head()

data.info()

print("Missing values:")
data.isnull().sum()

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
*/
```

## Output:
## Data head:
![image](https://github.com/user-attachments/assets/7bf86028-c222-47fe-ba50-53a0301161b0)
## Data Info:
![image](https://github.com/user-attachments/assets/1b810135-0ed3-4d26-b109-6ffcb0df94f7)
## Missig values:
![image](https://github.com/user-attachments/assets/c1992949-6330-48cf-8abe-775d4186b02e)
## Y_predicted value:
![image](https://github.com/user-attachments/assets/e1a440ae-518c-4406-bd11-ea85f7ce2733)
## Accuracy:
![image](https://github.com/user-attachments/assets/27e5c757-e02c-4d67-adf8-1bf6b32b397b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
