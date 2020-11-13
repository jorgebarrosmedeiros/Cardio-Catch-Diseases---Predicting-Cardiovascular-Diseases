import os
import pickle
import pandas as pd
import numpy as np

class Cardio(object):
    
    def __init__(self):
        
        #self.smt = pickle.load(smt, open("/home/jorge/repos/pa001_cardio_catch_diseases/parameter/smt.pkl", "rb"))
        self.scaler_ap_lo = pickle.load(open("parameter/rescaling_ap_lo.pkl","rb"))
        self.scaler_ap_hi = pickle.load(open("parameter/rescaling_ap_hi.pkl","rb"))
        self.scaler_weight = pickle.load(open("parameter/rescaling_weight.pkl","rb"))
        self.scaler_bmi = pickle.load(open("parameter/rescaling_bmi.pkl","rb"))
        self.scaler_age = pickle.load(open("parameter/rescaling_age_year.pkl","rb"))
        self.scaler_height = pickle.load(open("parameter/rescaling_height.pkl","rb"))

        
    def data_cleaning(self,df1):
    
        #coverting the age column to the years format
        df1['age_year'] = df1['age'].apply(lambda x: x/365)

        #age_year convert
        df1['age_year'] = df1['age_year'].astype(int)

        #changing id type to object
        df1['id'] = df1['id'].astype("object")
        
        return df1

    def feature_engineering(self, df2):

        #blood pressure
        df2['blood_pressure'] = df2.apply(lambda x:'normal' if (x['ap_hi'] <= 120) & (x['ap_lo'] <= 80)
                                         else 'elevated' if (x['ap_hi'] > 120 and x['ap_hi'] <= 129) & (x['ap_lo'] <= 80)
                                         else "high_blood_pressure" if (x['ap_hi'] >= 130 and x['ap_hi'] <= 139) & (x['ap_lo'] <= 90)
                                         else 'hypertension', axis = 1)

        #bmi calculation
        df2['bmi'] = round((df2['weight']/((df2['height'] * df2['height'])/10000)),2)

        #creating status_bmi 
        df2['status_bmi'] = df2['bmi'].apply(lambda x: 'underweight' if x <= 18.5
                                                  else 'normal' if (x > 18.5) & (x < 24.9) 
                                                  else 'overweight' if (x >= 25) & (x <= 29.9)
                                                  else 'obse' if (x >= 30) & (x<= 34.9) 
                                                  else 'extremely_obese')

        #creating age rage
        df2['age_range'] = df2['age_year'].apply(lambda x:"0-50" if x <= 50
                                                     else "50-65" if (x>50) & (x<=65)
                                                     else ">65")

        cols_drop = ['age','id']
        df2.drop(columns = cols_drop, axis = 1, inplace = True)

        #copying a new dataframe
        df2_clean = df2.copy() 

        #removing outliers
        df2_clean = df2_clean[~((df2_clean['ap_hi'] >= 220) | (df2_clean['ap_hi'] <= 90) 
                                | (df2_clean['ap_lo'] >= 150) | (df2_clean['ap_lo'] <= 65))]

        #filtering values
        df2_clean = df2_clean[(df2_clean['height'] > 140) & (df2_clean['height'] < 220)]
        
        df2 = df2_clean.copy()
        
        return df2

    def data_preparation(self, df5):        
        
        #blood_pressure - OrdinalEncoding
        dict_blood = {'normal':1, 'elevated':2,'high_blood_pressure':3, 'hypertension':4}
        df5['blood_pressure'] = df5['blood_pressure'].map(dict_blood)

        #status_bmi - OrdinalEncoding
        dict_bmi = {'underweight':1,'normal':2,'overweight':3,'obse':4,'extremely_obese':5}
        df5['status_bmi'] = df5['status_bmi'].map(dict_bmi)

        #age_range - OrdinalEncoding
        dict_age_range = {'50-65':2,'0-50':1}
        df5['age_range'] = df5['age_range'].map(dict_age_range)

        #gender - OneHotEncoding
        df5['gender_01'] = df5['gender'].apply(lambda x: 1 if x == 1 else 0)
        df5['gender_02'] = df5['gender'].apply(lambda x: 1 if x == 2 else 0 )

        #apply sampler
        #x_smt, y_smt = self.smt.fit_resample(df5, df5['cardio'])

        #join target variable
        #df_smt = x_smt
        #df_smt['cardio'] = y_smt

        #Re-encoding
        #blood_pressure - OrdinalEncoding
        #dict_blood = {1:'normal',2:'elevated',3:'high_blood_pressure',4:'hypertension'}
        #df_smt['blood_pressure'] = df_smt['blood_pressure'].map(dict_blood)

        #status_bmi - OrdinalEncoding
        #dict_bmi = {1:'underweight',2:'normal',3:'overweight',4:'obse',5:'extremely_obese'}
        #df_smt['status_bmi'] = df_smt['status_bmi'].map(dict_bmi)

        #age_range - OrdinalEncoding
        #dict_age_range = {2:'50-65',1:'0-50'}
        #df_smt['age_range'] = df_smt['age_range'].map(dict_age_range)

        #gender - OneHotEncoding
        #df_smt.drop(columns = ['gender_01','gender_02'],axis = 1, inplace = True)

        #df6 = df_smt.copy()

        #split data
        #X = df6.drop(['cardio',], axis = 1).copy()
        #y = df6['cardio']

        #X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)
        
        #ap_lo - RobustScaler
        df5['ap_lo'] = self.scaler_ap_lo.fit_transform(df5[['ap_lo']].values)
        #X_test['ap_lo'] = self.scaler_ap_lo.fit_transform(X_test[['ap_lo']].values)

        #ap_hi - RobustScaler
        df5['ap_hi'] = self.scaler_ap_hi.fit_transform(df5[['ap_hi']].values)
        #X_test['ap_hi'] = self.scaler_ap_hi.fit_transform(X_test[['ap_hi']].values)

        #weight - RobustScaler
        df5['weight'] = self.scaler_weight.fit_transform(df5[['weight']].values)
        #X_test['weight'] = self.scaler_weight.fit_transform(X_test[['weight']].values)

        #bmi - RobustScaler
        df5['bmi'] = self.scaler_bmi.fit_transform(df5[['bmi']].values)
        #X_test['bmi'] = self.scaler_bmi.fit_transform(X_test[['bmi']].values)

        #age_year - MinMaxScaler
        df5['age_year'] = self.scaler_age.fit_transform(df5[['age_year']].values)
        #X_test['age_year'] = self.scaler_age.fit_transform(X_test[['age_year']].values)

        #height - MinMaxScaler 
        df5['height'] = self.scaler_height.fit_transform(df5[['height']].values)
        #X_test['height'] = self.scaler_height.fit_transform(X_test[['height']].values)

        #blood_pressure - OrdinalEncoding
        dict_blood = {'normal':1, 'elevated':2,'high_blood_pressure':3, 'hypertension':4}
        df5['blood_pressure'] = df5['blood_pressure'].map(dict_blood)
        #X_test['blood_pressure'] = X_test['blood_pressure'].map(dict_blood)

        #status_bmi - OrdinalEncoding
        dict_bmi = {'underweight':1,'normal':2,'overweight':3,'obse':4,'extremely_obese':5}
        df5['status_bmi'] = df5['status_bmi'].map(dict_bmi)
        #X_test['status_bmi'] = X_test['status_bmi'].map(dict_bmi)

        #age_range - OrdinalEncoding
        dict_age_range = {'50-65':2,'0-50':1}
        df5['age_range'] = df5['age_range'].map(dict_age_range)
        #X_test['age_range'] = X_test['age_range'].map(dict_age_range)

        #gender - OneHotEncoding
        df5['gender_01'] = df5['gender'].apply(lambda x: 1 if x == 1 else 0)
        df5['gender_02'] = df5['gender'].apply(lambda x: 1 if x == 2 else 0 )
        #X_test['gender_01'] = X_test['gender'].apply(lambda x: 1 if x == 1 else 0)
        #X_test['gender_02'] = X_test['gender'].apply(lambda x: 1 if x == 2 else 0 )
        
        #features that will be really important to our model
        cols_selected = ['ap_hi','ap_lo','age_year','weight','height','blood_pressure','bmi','cholesterol']
        #creating a dataframe with selected columns
        #X_train_boruta = X_train[cols_selected].copy()
        #X_test_boruta = X_test[cols_selected].copy()
        
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
        
        #join pred into the original data
        original_data['prediction'] = pred
        
        return original_data.to_json(orient = 'records')
