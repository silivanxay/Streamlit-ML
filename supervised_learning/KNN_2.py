# import required modules and load data file (fruit_data_with_colors.txt)
import streamlit as st

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


st.write('Import fruit data with colors')
# Read data_ table into data frame format using pandas.
data = pd.read_excel('../Dataset/fruit_data_with_colors.xlsx')

# Initial Data Plot
st.dataframe(data.head())

# Create a mapping from fruit label value to fruit name to make results easier to interpret
# Create a mapping from fruit label value to fruit name to make results easier to interpret
target_fruits_label = dict(zip(data.fruit_label.unique(), data.fruit_name.unique()))

st.title('Mapping from fruit label value to fruit name')
st.dataframe(list(target_fruits_label.items()))

# Let x be the set of feature values and y as target value.
x = data[['mass', 'width', 'height']].values
y = data['fruit_label'].astype(int).values

# Use train_test_split function to break the data into default 75/25 % train and test data.
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

st.tile("Show train data and test data")
col1, col2 = st.columns(2)

col1.write('x_train')
col1.write(x_train)
col1.write('y_train')
col1.write(y_train)

col2.write('x_test')
col2.write(x_test)
col2.write('y_test')
col2.write(y_test)

# Data scientists usually select k=sqrt(n).
st.title("KNN Value")

k = st.slider("Choose value of K", min_value=1, max_value=10, key='k', value=5)

# Train the classifier by passing in the training set data in X_train, and the labels in y_train to the classifiers fit method.
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# test set points as input and compute the accuracy
st.write("KNN accuracy")
st.write(knn.score(x_test, y_test), )

st.title("For example")

# Customize fruit with mass, width and height to classify
mass = st.slider("Choose value of mass", min_value=1, max_value=400, key='k', value=20)
width = st.slider("Choose value of width", min_value=1, max_value=10, key='k', value=4)
height = st.slider("Choose value of height", min_value=1, max_value=10, key='k', value=6)

st.write("a small fruit with mass {} g, width {} cm, height {} cm".format(mass, width, height))

input = [mass, width, height]
st.write("Data Visualize")
# Plot fruit's 3 features and classifier to 3d plot
fig = px.scatter_3d(data, x='mass', y='width', z='height', color='fruit_label')
fig.add_trace(
    go.Scatter3d(x=[input[0]], y=[input[1]], z=[input[2]], name="Point2predict", mode='markers',
                 marker=dict(
                     color='green',
                 ),
                 showlegend=False),
)
st.plotly_chart(fig)

# If we ask the classifier to predict the label using the predict method.
fruit_prediction = knn.predict([input])
st.write('= {}'.format(target_fruits_label[fruit_prediction[0]]))



