# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)


# S10.2: Perform this activity in Sublime editor after adding the above code.
# Create a function 'prediction()' that accepts 'SepalLength', 'SepalWidth', 'PetalLength' and 'PetalWidth' as inputs and returns the species name.
@st.cache()
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth):
  species = svc_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"


st.title("Iris Flower Prediction App")
sepal_length = st.slider(label = "Sepal Length", min_value = 0.0, max_value = 10.0)
sepal_width = st.slider(label = "Sepal Width", min_value = 0.0, max_value = 10.0)
petal_length = st.slider(label = "Petal Length", min_value = 0.0, max_value = 10.0)
petal_width = st.slider(label = "Petal Width", min_value = 0.0, max_value = 10.0)

if st.button(label = "predict"):
	predict = prediction(sepal_length, sepal_width, petal_length, petal_width)
	st.write("The flower is", predict)
	st.write("Accuracy Score of our model is", score)