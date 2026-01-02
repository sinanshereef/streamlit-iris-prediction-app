
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd

from train_models import train_model
st.set_page_config(page_title='Iris prediction',layout="wide")
st.header('Iris Flower Prediction')
st.subheader('Upload, analyze, and predict')
file = st.file_uploader('Upload your Dataset', type=['csv'])
if file:
    df=pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df)
    st.subheader("Summary of the Dataset")
    st.write(df.describe(include='all'))
    st.subheader('Go to Side Bar for checking the #Missing Values Counts,Duplicate Count,Output value count')
    with st.sidebar:
        st.header("Analyses")
        st.subheader("Missing Value Count")
        st.write(df.isna().sum())
        st.subheader('Duplicate value Count')
        st.write(df.duplicated().sum())

    st.header('Select Target Column')
    target = st.selectbox('Select the Target Column', df.columns)

    st.subheader('Correction HEATMAP')
    fig,ax=plt.subplots(figsize=(6,4),dpi=100)
    sns.heatmap(df.corr(numeric_only=True),annot=True,ax=ax,cmap='coolwarm',square=True)
    st.pyplot(fig,use_container_width=False)

    numeric_columns=df.select_dtypes(include=np.number).columns
    if len(numeric_columns)>=2:
        st.subheader('Interactive Graph')
        fig2=px.scatter(df,x=numeric_columns[0],y=numeric_columns[1],color=target)
        st.plotly_chart(fig2)

    if st.button('Train Model'):
        (acc,missing_before,missing_after,duplicates_before,duplicates_after)=train_model(df,target)
        st.success('Model Trained successfully')
        st.write(f"Accuracy:**{acc*100:.2f}%**")

        st.subheader('Dataset Report')
        st.info(f"Missing Values before: [{missing_before}]")
        st.success(f"Missing Values after: [{missing_after}]")
        st.info(f"Duplicates Before: [{duplicates_before}]")
        st.success(f"Duplicates After: [{duplicates_after}]")

    model=None
    feature_encoder=None
    target_encoder=None

    try:
        model=pickle.load(open("model.pkl","rb"))
        feature_encoder=pickle.load(open("feature_encoder.pkl","rb"))
        target_encoder=pickle.load(open("target_encoder.pkl","rb"))
    except:
        pass

    if model is not None:
        st.subheader('Enter the Custom input of yours to make prediction')
        input_data=[]
        other_columns=df.drop(columns=[target])
        for col in other_columns.columns:
            if other_columns[col].dtype=='object':
                options=other_columns[col].unique().tolist()
                val=st.selectbox(col,options)
                val=feature_encoder[col].transform([val])[0]
            else:
                val=st.number_input(f"{col}",value=float(df[col].mean()))

            input_data.append(val)

        if st.button('Predict'):
            prediction=model.predict([input_data])

            if target_encoder:
                prediction=target_encoder.inverse_transform(prediction)

            st.success(f"Predicted Result:**{prediction[0]}**")

    else:
        st.warning("Please train the model 1st to enable the prediction")






