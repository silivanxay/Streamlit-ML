import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

st.write('Load dataset')
file = st.sidebar.file_uploader('Enter a file path:')
if file:
    data_full = pd.read_excel(file)
    data = data_full[["mpg", "wt", "drat", "qsec", "hp"]]
    st.write('summarize shape')
    st.write(data.shape)
    st.write('peak data')
    st.dataframe(data.head())

    # define predictor and response variables
    st.write('Train and test dataset creation')
    x = data[["mpg", "wt", "drat", "qsec"]]
    y = data["hp"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                        random_state=40)
    st.write('x_train and y_train')
    st.write(x_train.shape, y_train.shape)
    st.write('x_test and y_test')
    st.write(x_test.shape, y_test.shape)


    alpha = st.sidebar.slider("Select aplha", min_value=0.1, max_value=3.0, key='k',
                                 value=1.0, step=0.1)

    selection = st.sidebar.selectbox(
        'Which algorithm?',
        ('Ridge Regression', 'Lasso Regression')
    )
    algo_option = {
        'Ridge Regression': Ridge,
        'Lasso Regression': Lasso
    }
    st.title('Build the {} model with alpha'.format(selection))

    model = algo_option[selection](alpha=alpha)
    model.fit(x_train, y_train)
    pred_train_lasso = model.predict(x_train)
    pred_test_lasso = model.predict(x_test)

    # Plot car feature to 3d plot
    fig = px.scatter_3d(data, x='mpg', y='wt', z='hp', size='qsec',
                        color='drat', )

    x_test_sort = x_test.sort_values(by=['mpg'])
    y_pred_sort = model.predict(x_test_sort)

    fig.add_traces(go.Scatter3d(x=x_test_sort['mpg'],
                                y=x_test_sort['wt'],
                                z=y_pred_sort,
                                name='Regression Fit',
                                marker=dict(
                                  color='red', size=0.1
                                )))
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    st.plotly_chart(fig)


    st.subheader('Evaluate the lasso model')
    st.write('Mean squared error = {}'.format(
        np.sqrt(mean_squared_error(y_train,
                                   pred_train_lasso))))
    st.write('Accuracy = {}'.format(
        model.score(x_test, y_test), )
    )

    st.subheader('For example, define a new car with the following attributes:')
    st.sidebar.write('For example:')
    mpg = st.sidebar.slider("Choose value of mpg", min_value=1.0, max_value=30.0, key='k', value=24.0, step=0.1)
    wt = st.sidebar.slider("Choose value of wt", min_value=1.0, max_value=10.0, key='k', value=2.5, step=0.1)
    drat = st.sidebar.slider("Choose value of drat", min_value=1.0, max_value=10.0, key='k', value=3.5, step=0.1)
    qsec = st.sidebar.slider("Choose value of qsec", min_value=1.0, max_value=25.0, key='k', value=18.5, step=0.1)

    st.write('value of mpg = {}'.format(mpg))
    st.write('value of wt = {}'.format(wt))
    st.write('value of drat = {}'.format(drat))
    st.write('value of qsec = {}'.format(qsec))

    # define new observation
    new = [mpg, wt, drat, qsec]

    # predict hp value using Ridge/lasso regression model
    example_prediction = model.predict([new])
    st.markdown('predicted value of hp = __{}__'.format(example_prediction[0]))
