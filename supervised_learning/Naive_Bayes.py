import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


file = st.file_uploader('Enter a file path:')
if file:
    data = pd.read_excel(file)
    # Initial Data Plot
    st.dataframe(data.head())

    # Let x be the set of feature values and y as target value.
    x = data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].values
    y = data['Species'].values

    # Create the X, Y, Training and Test
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.60,  random_state=0)

    st.title("Show train data and test data")
    col1, col2 = st.columns(2)

    col1.write('x_train')
    col1.write(x_train)
    col1.write('y_train')
    col1.write(y_train)

    col2.write('x_test')
    col2.write(x_test)
    col2.write('y_test')
    col2.write(y_test)

    # Train the classifier by passing in the training set data in X_train,
    # and the labels in y_train to the classifiers fit method.
    model=GaussianNB()
    model.fit(x_train, y_train)


    # test set points as input and compute the accuracy
    st.write("Naive Bayes accuracy")
    st.write(model.score(x_test, y_test), )



    st.title("For example")

    # Customize flower with [Sepal.Length, Sepal.Width, Petal.Length, Petal.Width] to classify
    Sepal_Length = st.slider("Choose value of SepalLength", min_value=1.0, max_value=8.0, key='k', value=5.0, step=0.1)
    Sepal_Width = st.slider("Choose value of Sepal.Width", min_value=2.0, max_value=5.0, key='k', value=4.0, step=0.1)
    Petal_Length = st.slider("Choose value of Petal.Length", min_value=1.0, max_value=6.0, key='k', value=6.0, step=0.1)
    Petal_Width = st.slider("Choose value of Petal.Width", min_value=0.0, max_value=3.0, key='k', value=2.0, step=0.1)

    input = [Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]

    st.write("a small flower with SepalLength {}, Sepal.Width {}, Petal.Length {}, Petal.Width {}"
             .format(Sepal_Length, Sepal_Width, Petal_Length, Petal_Width))

    st.write("Data Visualize")
    # Plot fruit's 3 features and classifier to 3d plot
    fig = px.scatter_3d(data, x='Sepal.Length', y='Sepal.Width', z='Petal.Length', size= 'Petal.Width', color='Species', )
    fig.add_trace(
        go.Scatter3d(x=[input[0]], y=[input[1]], z=[input[2]], name="Point2predict", mode='markers',
                     marker=dict(
                         color='red',
                         size=input[3]
                     ),
                     showlegend=True),
    )
    st.plotly_chart(fig)

    # If we ask the classifier to predict the label using the predict method.
    example_prediction = model.predict([input])
    st.write('= {}'.format(example_prediction[0]))


