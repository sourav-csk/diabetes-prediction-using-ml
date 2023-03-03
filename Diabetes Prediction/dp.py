import pickle
import pandas as pd   
import numpy as np   
import matplotlib.pyplot as plt         
import plotly_express as px    
from PIL import Image
import seaborn as sns 
import streamlit as st 
import altair as alt
import random
from streamlit_option_menu import option_menu 




#loading the saved model 

diabetes_model = pickle.load(open('D:/Exposys Data Lab/Diabetes Prediction/saved model/diabetes_model1.sav','rb'))


#sidebar for navigation
with st.sidebar:
    
    selected = option_menu(
               menu_title="Diabetes Disease Prediction System",
                           
                          
                          options=['Diabetes Prediction'],
                          icons=['person'],
                          orientation='horizontal',
                          default_index=0)
    

# Diabetes Prediction Page....

if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
        
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
    
    
# creating a button for graphical Prediction
def main():
 if st.button('Diabetes Test Graphical Result'):
  st.title('Visualization plots using DataSet')
  diabetes_dataset= pd.read_csv('D:\Exposys Data Lab\Diabetes Prediction\Dataset\diabetes.csv')
  st.dataframe(diabetes_dataset)
  
#using pyplot function
  st.title('Visualization plots using Age DataSet')
  fig1 = plt.figure()
  diabetes_dataset['Age'].value_counts().plot(kind= 'bar')
  st.pyplot(fig1)


  st.title('Visualization plots using Outcome DataSet')
  fig2 = plt.figure()
  sns.countplot(x='Outcome', data=diabetes_dataset)
  st.pyplot(fig2) 
  
  fig =plt.figure()
  st.title('Visualization plots using BloodPressure DataSet')
  sns.countplot(x='BloodPressure', data=diabetes_dataset)
  st.pyplot(fig)  
  
  st.title('Visualization plots using BMI DataSet')
  fig3= plt.figure()
  diabetes_dataset['BMI'].value_counts().plot(kind= 'kde')
  st.pyplot(fig3) 
    
  st.title('Visualization plots using Pregnancies DataSet')
  st.bar_chart(diabetes_dataset[['Pregnancies','Age']])
  
  st.dataframe(diabetes_dataset.head())
  
  st.title('Visualization plots using Pregnancies,Age,BMI DataSet')
  st.line_chart(diabetes_dataset[['Pregnancies','Age','BMI']])
  
  st.area_chart(diabetes_dataset[['Pregnancies','Age','BMI']])
  
  #using
main()
#st.success()   


