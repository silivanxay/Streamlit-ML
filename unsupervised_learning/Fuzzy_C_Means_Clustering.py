import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from fcmeans import FCM
from sklearn.metrics import accuracy_score


st.header('Fuzzy C-Means Clustering')
file = st.sidebar.file_uploader('Enter a file path:')
if file:
    data = pd.read_excel(file)
    st.write('Peek into Data')
    st.dataframe(data.head())

    # Let x be the set of feature values and y as target value.
    x = data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].values


    # Train the clustering by passing in the training set data in X.
    model = FCM(n_clusters=3)
    model.fit(x)

    always_virginca = np.array([7.7, 2.6, 6.9, 2.3])
    always_setosa = np.array([5.7, 4.4, 1.5, 0.4])
    always_versicolor = np.array([5.1, 2.5, 3, 1.1])

    label_versicolor = model.predict(always_virginca)[0].item()
    label_setosa = model.predict(always_setosa)[0].item()
    label_virginica = model.predict(always_versicolor)[0].item()

    st.write('Mapping values to species')
    mapping_label = {
        label_versicolor: 'versicolor',
        label_setosa: 'setosa',
        label_virginica: 'virginica',
    }
    st.write(mapping_label)

    y = data['Species'].values
    # outputs
    y_predict = [mapping_label[value] for value in model.predict(x)]

    accuracy = accuracy_score(y, y_predict)
    st.subheader('Accuracy = {}'.format(accuracy))

    predict_data = data.copy()
    # Drop that column
    predict_data.drop('Species', axis=1,  inplace=True)

    # Put whatever series you want in its place
    predict_data['Species'] = y_predict


    st.title("For example")

    # Customize flower with [Sepal.Length, Sepal.Width, Petal.Length, Petal.Width] to classify
    Sepal_Length = st.sidebar.slider("Choose value of SepalLength", min_value=1.0, max_value=8.0, key='k', value=5.0, step=0.1)
    Sepal_Width = st.sidebar.slider("Choose value of Sepal.Width", min_value=2.0, max_value=5.0, key='k', value=4.0, step=0.1)
    Petal_Length = st.sidebar.slider("Choose value of Petal.Length", min_value=1.0, max_value=6.0, key='k', value=6.0, step=0.1)
    Petal_Width = st.sidebar.slider("Choose value of Petal.Width", min_value=0.0, max_value=10.0, key='k', value=2.0, step=0.1)

    input = np.array([Sepal_Length, Sepal_Width, Petal_Length, Petal_Width])

    st.write("a small flower with SepalLength {}, Sepal.Width {}, Petal.Length {}, Petal.Width {}"
             .format(Sepal_Length, Sepal_Width, Petal_Length, Petal_Width))

    st.subheader("Data Visualize")
    # Plot fruit's 3 features and classifier to 3d plot
    fig = px.scatter_3d(predict_data, x='Sepal.Length', y='Sepal.Width', z='Petal.Length', size='Petal.Width',
                        color='Species', )
    fig.add_trace(
        go.Scatter3d(x=[input[0]], y=[input[1]], z=[input[2]], name="Point2predict", mode='markers',
                     marker=dict(
                         color='red',
                         size=input[3]
                     ),
                     showlegend=True),
    )
    st.plotly_chart(fig)

    # If we ask the clustering to predict the label using the predict method.
    example_prediction = mapping_label[model.predict(input)[0]]
    if st.sidebar.button('Predict'):
        st.subheader('= {}'.format(example_prediction))
