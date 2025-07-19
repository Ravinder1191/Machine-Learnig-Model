import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

st.title("Iris Flower Classification using Logistic Regression")
st.write("This app helps you classify iris flowers based on their features using a machine learning model.")

iris = load_iris()
feature = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.Series(iris.target)

if st.button("Preview Sample Data"):
    st.write("Here are the first few records from the dataset:")
    st.dataframe(feature.head())

if st.button("Split Data into Train and Test Sets"):
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)
    st.session_state['x_train'] = x_train
    st.session_state['x_test'] = x_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.success("Data has been split into training and testing sets.")


if st.button("Train the Logistic Regression Model"):
    if 'x_train' in st.session_state:
        model = LogisticRegression(max_iter=200)
        model.fit(st.session_state['x_train'], st.session_state['y_train'])
        st.session_state['model'] = model
        st.success("Model has been trained successfully.")
    else:
        st.warning("Please split the data first.")


if st.button("Predict Test Data"):
    if 'model' in st.session_state:
        predictions = st.session_state['model'].predict(st.session_state['x_test'])
        st.session_state['predictions'] = predictions
        st.success("Predictions on test data are ready.")
    else:
        st.warning("Train the model before making predictions.")


if st.button("Show Confusion Matrix"):
    if 'predictions' in st.session_state:
        conf_matrix = confusion_matrix(st.session_state['y_test'], st.session_state['predictions'])
        st.write("The confusion matrix below shows how well the model performed:")
        st.write(f"accuracy: {conf_matrix.accuracy}")
        st.dataframe(conf_matrix)
    else:
        st.warning("Run the prediction step first.")


if st.checkbox("Show Training Data"):
    if 'x_train' in st.session_state:
        st.write("This is a portion of the data used to train the model:")
        st.dataframe(st.session_state['x_train'].head())
    else:
        st.warning("Split the data first to view this.")
