

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def train_model(df,target):
    missing_values_before=df.isna().sum()
    duplicates_before=df.duplicated().sum()
    df=df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype=='object':
            df[col]=df[col].fillna(df[col].mode()[0])
        else:
            df[col]=df[col].fillna(df[col].mean())

    missing_values_after=df.isna().sum()
    duplicates_after=df.duplicated().sum()

    x=df.drop(columns=[target]) #x,y split
    y=df[target]
    feature_encoder={}

    for col in x.columns:  #encoding
        if x[col].dtypes=='object':
            le=LabelEncoder()
            x[col]=le.fit_transform(x[col])
            feature_encoder[col]=le

    target_encoder=None

    if y.dtypes=='object':
        target_encoder=LabelEncoder()
        y=target_encoder.fit_transform(y)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1) #train_test_split
    model=DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train,y_train)
    accuracy=model.score(x_test,y_test)
    pickle.dump(model,open("model.pkl","wb"))
    pickle.dump(feature_encoder,open('feature_encoder.pkl','wb'))
    pickle.dump(target_encoder,open('target_encoder.pkl','wb'))
    return (
        accuracy,
        missing_values_before,
        missing_values_after,
        duplicates_before,
        duplicates_after
    )

