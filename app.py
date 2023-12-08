#streamlit app for predicting 

#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import pickle
from sklearn.metrics import classification_report

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#App title
st.title('Investment Avenue Prediction')

#add an upload button for data upload
uploaded_file = st.file_uploader("Upload your input csv file", type=["csv"])

#add a progress bar
import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1)

#create a dataframe from uploaded csv file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    #get input from user on muber of rows to display
    num_rows = st.number_input('Enter the number of rows to display', min_value=0, max_value=30, value=5)
    #show the top 5 rows of the dataframe
    st.header("Data Sample")
    st.dataframe(data.head(num_rows))

#create a function to plot categorical variables
def plot_cat(data, cat_var):
    st.header("Plot of " + cat_var)
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.countplot(data=data, x=cat_var)
    plt.title(cat_var)
    plt.show()
    st.pyplot(fig)

#get a list of all columns in the dataframe
columns = data.columns.tolist()

#create a dropdown where a user can select the column to plot
cat_var = st.selectbox('Select a column to plot', columns)

#plot the selected column
plot_cat(data, cat_var)

# create a function to encode categorical variables
def encode_cat(data, cat_var):
    encoder = OrdinalEncoder()
    data[cat_var] = encoder.fit_transform(data[[cat_var]])
    return data

# for loop to encode all categorical variables
for i in data.columns:
    if data[i].dtypes == 'object':
        encode_cat(data, i)

#show the top 3 rows of our updated dataframe
st.header("Data Encoded Dataframe")
st.dataframe(data.head(3))

#split the data uploaded into features and target
x= data.drop(columns=['Avenue'])

#import the model
model = pickle.load(open('model.pkl', 'rb'))

#make predictions using the model
prediction = model.predict(x)

#add the predictions to the dataframe
data['Avenue_prediction'] = prediction

#get user input on the number of rows to display
num_rows_pred = st.number_input('Enter the number of rows to display', min_value=0, max_value=20, value=5)

#show our top 5 rows of the dataframe
st.header("Predictions")
st.dataframe(data.head(num_rows_pred))

#print the classification report
st.header("Classification Report")
st.text("0 = Mutual Fund, 1 = Equity, 2 = Fixed Deposits, 3 =Public Provident Fund")

class_report = classification_report(data['Avenue'], data['Avenue_prediction'])
st.text(class_report)