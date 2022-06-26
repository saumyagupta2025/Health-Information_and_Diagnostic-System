from turtle import onclick, width
from xml.etree.ElementTree import SubElement
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import requests
from streamlit_option_menu import option_menu
from PIL import Image
import sys 
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image
#import SessionState


hide_menu="""
<style>
#MainMenu{
    visibility:visible;
}
footer{
    visibility:visible;
    text-align: center
    display:block;
    position:relative;
    padding:5px;
    top:3px;
}
footer:after{
    content:'Built & developed by Saumya Gupta';
    display:block;
    position:relative;
    color:tomato;
    text-align: center
    padding:5px;
    top:3px;
}
footer:before{
    content:'Disclaimer: This content is developed to demonstrate the capabilities of Machine Learning in the field of Healthcare. The text, graphics, images, and content, used is general in nature and for informational purposes only and does not constitute medical advice; the content is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always consult your doctor or other qualified healthcare provider with any questions you may have regarding a medical condition.';
    display:block;
    position:relative;
    color:grey;
    text-align:center;
    padding:5px;
    top:3px;
}
</style>
"""

# Remove whitespace from the top of the page and sidebar
    
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False



def callback():
    st.session_state.button_clicked = True


def symptoms():
    st.title("Disease Prediction based on Symptoms")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/Cough-removebg-preview.png")
        st.image("Cough1-removebg-preview.png", width = 100,caption = "Cough")
    with col2:
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/Fever-removebg-preview.png")
        st.image("Fever1-removebg-preview.png", width = 100,caption = "Fever")    
    with col3:
        st.image("MuscleAche1-removebg-preview.png", width = 100, caption= "Muscle or Body Ache")
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/MuscleAche-removebg-preview.png")
    with col4:
        st.image("RunnyNose1-removebg-preview.png", width = 100,caption = "Congestion or Runny Nose")
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/RunnyNose-removebg-preview.png")
    with col5:
        st.image("Sorethroat1-removebg-preview.png", width = 100,caption = "Sore Throat")
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/SoreThroat-removebg-preview.png")
    with col6:
        st.image("ShortnessOfBreadth-removebg-preview.png", width = 100,caption = "Shortness of Breadth")


    symptoms_list = pickle.load(open("symptoms_list.pkl", "rb"))
    options = st.multiselect("Select your symptoms", symptoms_list)
    
    symptoms_df = pd.DataFrame(columns=symptoms_list)
    #print(symptoms_df)
    row = []
    for i in range(0, len(symptoms_list)):
        row.append(0)
    #print(len(row))
    #print(len(symptoms_list))
    #print(options)
    for i in range(0, len(symptoms_list)):
        for option in options:
            if  symptoms_list[i] == option:
                row[i] = 1
            else:
                row[i] = 0
    #print(row)
    model_symptoms = pickle.load(open("Symptoms_model.pkl", "rb"))
    encoder = pickle.load(open("labelEncoder_for_Symptoms.pkl", "rb"))
    
    if st.button('Predict'):   
        if(options == []):
            st.error("No symptoms selected") 
        
        else:
            row = np.array(row).reshape(1, -1)
        
        #test_df = symptoms_df.append(row, ignore_index=True)
            prediction = model_symptoms.predict(row)
            result = encoder.inverse_transform(prediction)
        #res = str("You may have ") +str(result)
        

            st.success("You may have " + result)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")


def stroke_home():
    st.title("Brain Stroke Prediction System")
    image = Image.open("Stroke.jpeg")
    st.write("")
    st.image(image, caption = None, width = 700)

    #st.subheader("What is Stroke?")
    st.write("")
    
    '''
    Stroke is the third leading cause of death and the leading cause of adult disability. A stroke is an interruption of the blood supply to any part of the brain. It is also referred to as a brain attack. If blood flow was stopped for longer than a few seconds and the brain cannot get blood and oxygen, brain cells can die, and the abilities controlled by that area of the brain are lost.
    
    '''


    st.subheader("There are two types of strokes: Ischemic Stroke and Hemorrhagic Stroke")
    '''
    An ischemic stroke is caused by a blood clot that blocks an artery and cut off blood flow to the brain. A hemorrhagic stroke is caused by the breakage or 'blowout' of a blood vessel in the brain. The ischemic stroke is by far the most common type of stroke.
    '''
    
    
    st.subheader("Stroke Symptoms")
    '''
    The symptoms of Stroke include: 
    
    * Sudden numbness or weakness of the face, arm or leg, especially on one side of the body
    * Sudden confusion, trouble speaking or understanding
    * Sudden trouble seeing in one or both eyes
    * Sudden trouble walking, dizziness, loss of balance and coordination
    * Sudden severe headache with no known cause

   '''
    st.write("")
    st.write("")
   
    if (st.button('Make prediction', key = 'stroke', on_click = callback) or st.session_state.button_clicked):
        st.write("")
        stroke()
        

def diabetes_home():
    #callback1()
    st.title("Diabetes Prediction System")
    image = Image.open("Diabetes.jpeg")
    st.write("")
    st.image(image, caption = None, width = 700)

    st.write("")
    st. markdown("<p style='text-align: justify;'> Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored or used for energy. With diabetes, your body either doesn’t make enough insulin or can’t effectively use the insulin it does make.</p>", unsafe_allow_html=True)
    '''
    Untreated high blood sugar from diabetes can damage your nerves, eyes, kidneys, and other organs.
    '''

    st.subheader("Types of Diabetes")

    '''
    
    There are a few different types of diabetes:

    * Type 1 diabetes is an autoimmune disease. The immune system attacks and destroys cells in the pancreas, where insulin is made. It’s unclear what causes this attack. About 10 percent of people with diabetes have this type.
    * Type 2 diabetes occurs when your body becomes resistant to insulin, and sugar builds up in your blood.
    * Prediabetes occurs when your blood sugar is higher than normal, but it’s not high enough for a diagnosis of type 2 diabetes.
    * Gestational diabetes is high blood sugar during pregnancy. Insulin-blocking hormones produced by the placenta cause this type of diabetes.
    
    '''

    st.subheader("Symptoms of diabetes")
    '''
    The general symptoms of diabetes include:

    * Increased hunger
    * Increased thirst
    * Weight loss
    * Frequent urination
    * Blurry vision
    * Extreme fatigue
    * Sores that don’t heal
    
    '''

    st.write("")
    if((st.button('Make Predictions',  key = 'diabetes', on_click = callback) or st.session_state.button_clicked)):
        st.write("")
        diabetes()

def stroke():
    model = pickle.load(open("Stroke_rf_model.pkl", "rb"))
    labelEncoder = pickle.load(open("LabelEncoder.pkl","rb"))

    gender = st.selectbox("Enter the gender", ('Male', 'Female'))
    Age = st.number_input("Enter age ", min_value = 0, max_value = 100)
    Hypertension = st.selectbox("Do you have hypertension?", ('Yes', 'No'))
    Heart_disease = st.selectbox("Do you have heart disease?", ('Yes', 'No'))
    ever_married = st.selectbox("Are you married?", ('Yes', 'No'))
    smoking_status = st.selectbox("Select smoking status", ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))
    work_type = st.selectbox("Select work type", ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    Residence_type = st.selectbox("Select residence type", ('Urban' ,'Rural'))
    col1, col2 = st.columns(2)
    with col1:
        avg_glucose_level = st.number_input("Enter average glucose level")
    with col2:
        bmi = st.number_input("Enter BMI")
    if Hypertension == "Yes": 
        Hypertension = 1
    else: 
        Hypertension = 0

    if Heart_disease == "Yes": 
        Heart_disease = 1
    else: 
        Heart_disease = 0

    smoking_status = labelEncoder.fit_transform(np.array(smoking_status).reshape(-1,1))
    gender = labelEncoder.fit_transform(np.array(gender).reshape(-1,1))
    ever_married = labelEncoder.fit_transform(np.array(ever_married).reshape(-1,1))
    Residence_type = labelEncoder.fit_transform(np.array(Residence_type).reshape(-1,1))
    work_type = labelEncoder.fit_transform(np.array(work_type).reshape(-1,1))

    temp = np.concatenate((smoking_status, gender, ever_married, Residence_type, work_type), axis = 0)
    ##print(temp)
    if st.button('Predict'):
        test = [temp[1], Age, Hypertension, Heart_disease, temp[2],temp[4], temp[3], avg_glucose_level, bmi, temp[0]]
        test = np.array(test).reshape(1,-1)

        prediction = model.predict(test)
        #prediction
        if prediction == 1:
            st.error("You are at a risk of brain stroke!")

        else: 
            st.success("You are not at a risk of brain stroke.")
        

def diabetes():
    #st.title("Diabetes Prediction System")
    model_diabetes = pickle.load(open("diabetes_model_RandomForest.pkl", "rb"))

    Age = st.number_input("Enter age ", min_value = 0, max_value = 100)
    bMI_d = st.number_input("Enter BMI", min_value = 0.0, max_value = 60.0, format="%.2f")
    glucose_level = st.number_input("Enter average glucose level", format="%.2f")
    bP = st.number_input("Enter Blood Pressure", min_value = 30, max_value = 250)
    insulin = st.number_input("Enter insulin levels", min_value = 0, max_value = 300)
    pregnancies = st.number_input("Enter number of pregnancies", min_value = 0, max_value = 15)
    SkinThickness = st.number_input("Enter skin thickness", format="%.2f")
    DiabetesPedigreeFunction = st.number_input("Enter likelihood of diabetes based on family history", format="%.3f")

    sample = [pregnancies, glucose_level, bP, SkinThickness, insulin, bMI_d, DiabetesPedigreeFunction, Age]
    sample = np.array(sample).reshape(1, -1)
   
    if st.button('Predict'):

        pred = model_diabetes.predict(sample)
       
        if pred == 1:
            st.error("You may have diabetes!")
        else: 
            st.success("You are not at a risk of diabetes.")


def heartAttackHome():
    st.title("Heart Attack Prediction System")
    image = Image.open("heartAttackImage.jpeg")
    st.write("")
    st.image(image, caption = None, width = 700)

    st.markdown("<p style='text-align: justify;'> A heart attack occurs when the flow of blood to the heart is severely reduced or blocked. The blockage is usually due to a buildup of fat, cholesterol and other substances in the heart (coronary) arteries. The fatty, cholesterol-containing deposits are called plaques. The process of plaque buildup is called atherosclerosis.</p>", unsafe_allow_html = True)
    '''
    
    Sometimes, a plaque can rupture and form a clot that blocks blood flow. A lack of blood flow can damage or destroy part of the heart muscle. A heart attack is also called a myocardial infarction.
    
    '''


    st.subheader("Symptoms of Heart Attack")
    '''
    Symptoms of a heart attack vary. Some people have mild symptoms. Others have severe symptoms. Some people have no symptoms.

    Common heart attack symptoms include:

    * Chest pain that may feel like pressure, tightness, pain, squeezing or aching
    * Pain or discomfort that spreads to the shoulder, arm, back, neck, jaw, teeth or sometimes the upper belly
    * Cold sweat
    * Fatigue
    * Heartburn or indigestion
    * Lightheadedness or sudden dizziness
    * Nausea
    * Shortness of breath
    
    '''
    st.write("")

    if((st.button('Make Predictions',  key = 'HeartAttack', on_click = callback) or st.session_state.button_clicked)):
        st.write("")
        heartAttack()
    






def heartAttack():
    model = pickle.load(open("Heart_attack_model_KNN.pkl", "rb"))
    Age = st.number_input("Enter age ", min_value = 0, max_value = 120)
    gender = st.selectbox("Enter the gender", ('Male', 'Female'))
    
    Chest_Pain_Type = st.selectbox("Enter chest pain type:",(0, 1, 2, 3))
    st.text("Value 0: typical angina\nValue 1: atypical angina\nValue 2: non-anginal pain\nValue 3: asymptomatic")
    trtbps = st.number_input("Resting blood pressure (in mm Hg)")
    chol  = st.number_input("Cholestoral in mg/dl fetched via BMI sensor")
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", (0,1))
    rest_ecg = st.selectbox("Resting electrocardiographic results", ('Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy by Estes criteria'))
    thalach = st.number_input("Maximum Heart Rate")
    exng = st.selectbox("Exercise induced angina",('Yes', 'No'))
    caa = st.selectbox("Number of major vessels", (0,1,2,3))
    oldpeak = st.number_input("Previous Peak")
    slp = st.selectbox("Slope", (0,1,2))
    thall = st.selectbox("Thall rate", (0,1,2,3))

    if exng == 'Yes':
        exng = 1
    else:
        exng = 0;

    if(rest_ecg == 'Normal'):
        rest_ecg = 0
    elif(rest_ecg == 'Having ST-T wave abnormality'):
        rest_ecg = 1
    else: 
        rest_ecg = 2
    
    if gender == 'Male':
        gender = 1
    else : 
        gender = 0

    sample =[Age, gender, Chest_Pain_Type, trtbps, chol, fbs, rest_ecg, thalach, exng, oldpeak, slp, caa, thall]
    sample = np.array(sample).reshape(1,-1)
    if st.button('Predict'):
        pred = model.predict(sample)

        if pred == 1:
            st.error("You are at a risk of heart attack!")
        else:
            st.success("You are not at a risk of heart attack.")


def TbHome():
    st.title("Prediction of Tuberculosis using Chest X-Ray Images")
    descImage = Image.open("Tuberculosis.jpeg")
    st.image(descImage, caption=None, width=700)

    st.write("")

    '''
    Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. The bacteria that cause tuberculosis are spread from person to person through tiny droplets released into the air via coughs and sneezes.
    '''

    st.subheader("Symptoms of Tuberculosis")
    '''
    Signs and symptoms of active TB include:

    * Coughing for three or more weeks
    * Coughing up blood or mucus
    * Chest pain, or pain with breathing or coughing
    * Unintentional weight loss
    * Fatigue
    * Fever
    * Night sweats
    * Chills
    * Loss of appetite
    '''
    st.markdown("<p style='text-align: justify;'>Tuberculosis can also affect other parts of your body, including the kidneys, spine or brain. When TB occurs outside your lungs, signs and symptoms vary according to the organs involved. For example, tuberculosis of the spine might cause back pain, and tuberculosis in your kidneys might cause blood in your urine.</p>", unsafe_allow_html = True)
    
    st.write("")

    if (st.button('Upload Image', key='Tuberculosis', on_click=callback) or st.session_state.button_clicked):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        predict_tuberculosis()


def predict_tuberculosis():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    fp = "cnn_tuberculosis_model.h5"
    cnn = load_model(fp)
    # @st.cache(allow_output_mutation=True)
    #cnn = loading_model()

    #cnn = load_model("/content/cnn_tuberculosis_model.h5")

    upload = st.file_uploader("Upload X-Ray Image")

    buffer = upload
    temp_file = NamedTemporaryFile(delete=False)
    if buffer:
        temp_file.write(buffer.getvalue())
        st.write(image.load_img(temp_file.name))

    if buffer is None:
        st.text("Please upload a image")

    else:
        my_img = image.load_img(temp_file.name, target_size=(
            500, 500), color_mode='grayscale')

        # Preprocessing the image
        prep_img = image.img_to_array(my_img)
        prep_img = prep_img/255
        prep_img = np.expand_dims(prep_img, axis=0)

        # predict
        preds = cnn.predict(prep_img)
        if preds >= 0.5:
            out = ('I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(
                preds[0][0]))

        else:
            out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(
                1-preds[0][0]))

        st.success(out)

        tempImage = Image.open(temp_file)
        st.image(tempImage, use_column_width=True)



def welcome():
    
    st.title("Health Information & Diagnostic System")
    st.image("dbwh8kv-375c5c96-00bc-4bd7-b57a-b9908074ed18.jpeg", width = 700)
    st.write("")
    st. markdown("<p style='text-align: justify;'>Good health is central to human happiness and well-being that contributes significantly to prosperity and wealth and even economic progress, as healthy populations are more productive, save more and live longer. When a person leads a healthy lifestyle, the body remains healthy and the mind is active and fresh. The most important concept of leading a healthy life is to provide immunity strength against various diseases for which timely detection and identification of diseases is a necessity. The traditional methods which are used to diagnose a disease are manual and error-prone. Usage of Artificial Intelligence (AI) predictive techniques enables auto diagnosis and reduces detection errors compared to exclusive human expertise.  Disease diagnosis is the identification of an health issue, disease, disorder, or other condition that a person may have. Disease diagnoses could be sometimes very easy tasks, while others may be a bit trickier. Disease detection driven by artificial intelligence (AI) has demonstrated to be an effective tool for identifying undiagnosed patients with complex common as well as rare diseases. The use of these algorithms is driven by awareness that underdiagnosis leads to a heavy burden for patients and healthcare professionals, and is also a challenge for pharmaceutical companies seeking to expand the patient pool for their medications, whether to power clinical trials or to efficiently target healthcare providers.</p>", unsafe_allow_html=True) 



def __main__():

    
    st.markdown(hide_menu, unsafe_allow_html=True) 
    with st.sidebar:
        #st.title("Menu")
        selected = option_menu(menu_title = "Menu", 
                            options = ["Home",  "Disease Prediction based on Symptoms" ,"Prediction of Brain Stroke", "Prediction of Diabetes", "Prediction of Heart Attack", "Prediction of Tuberculosis"], 
                            default_index=0, 
                            menu_icon=None, 
                            icons=None, 
                            orientation="vertical",
                            styles=None, 
                            key=None)
   

    if selected == 'Prediction of Brain Stroke':
        stroke_home()
    if selected == 'Prediction of Diabetes':
        diabetes_home()
    if selected == 'Prediction of Heart Attack':
        heartAttackHome()
    if selected == 'Home':
        welcome()
    if selected == 'Disease Prediction based on Symptoms':
        symptoms()
    if selected == 'Prediction of Tuberculosis':
        TbHome()
        


__main__()