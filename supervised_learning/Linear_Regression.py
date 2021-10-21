import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file = st.sidebar.file_uploader('Please upload file')
if file:
    x_label = "YearsExperience"
    y_label = "Salary"
    data = pd.read_excel(file)
    # Initial Data Plot
    st.write('Peek data')
    st.dataframe(data.head())

    st.write("Generating the scatter plot")
    fig = px.scatter(data_frame=data, y=y_label, x=x_label)
    st.plotly_chart(fig)

    # Let x be the set of feature values and y as target value.
    x = data[x_label].to_numpy().reshape(-1, 1)
    y = data[y_label]

    # reshape x data to 2d because linear regression function required
    st.write('Reshape x data from 1d to 2d')
    st.write(x)

    # Create the X, Y, Training and Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.30, random_state=0)

    st.title("Show train data and test data")
    col1, col2 = st.columns(2)
    container1 = st.container()
    container2 = st.container()

    with container1:
        col1.write('x_train')
        col1.write(x_train)
        col1.write('y_train')
        col1.write(y_train)

    with container2:
        col2.write('x_test')
        col2.write(x_test)
        col2.write('y_test')
        col2.write(y_test)

    # Train the regression by passing in the training set data in X_train,
    # and the labels in y_train to the regression fit method.
    model = LinearRegression()
    model.fit(x_train, y_train)

    # test set points as input and compute the intercept and slope
    st.subheader('Train the regression')
    st.write("Intercept = ", model.intercept_)
    st.write("Slope = ", model.coef_[0])

    # test set points as input and compute the accuracy
    st.write("Linear regression accuracy")
    st.write(model.score(x_test, y_test), )

    st.write('Actual vs. Predicted values:')
    y_pred = model.predict(x_test)
    comparing = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe (comparing)

    # generating the scatter plot
    st.write("Generating the scatter plot")
    fig = px.scatter(x=x_test.flatten(), y=y_test,
                     labels={'x': x_label, 'y': y_label})
    fig.add_trace(
        go.Scatter(x=x_test.flatten(), y=y_pred, name="predict",  mode='markers',
                   marker=dict(
                       color='red',
                   ), )
    )

    x_range = np.linspace(x_test.flatten().min(), x_test.flatten().max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit',
                              marker=dict(
                                    color='red',)
                              ,),)
    st.plotly_chart(fig)

    st.subheader("For example")
    # Customize number of year
    num_year = st.sidebar.slider("For example: Choose value of number of year", min_value=1.0, max_value=20.0, key='k', value=11.0, step=0.1)
    st.write("with number of year _{}_".format(num_year))
    input = [[num_year]]
    example_prediction = model.predict(input).tolist()
    st.markdown('= __{}__'.format(example_prediction[0]))

