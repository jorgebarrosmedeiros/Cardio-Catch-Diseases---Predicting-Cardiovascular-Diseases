import os
import pickle
from flask import Flask, request, Response
import pandas as pd
from cardio.Cardio import Cardio

model = pickle.load(open("/model/model_cardio_lgbm.pkl", "rb"))

app = Flask(__name__)

@app.route("/cardio/predict", methods=['POST'])
def cardio_predict():
    print("Aqui")	
    test_json = request.get_json(force = True)
    print(test_json)
    
    if test_json:
        if isinstance(test_json,dict):           
            test_raw = pd.DataFrame(test_json, index = [0])
        else:
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
            
        pipeline = Cardio()
        print(test_raw.head())
        
        #data cleaning
        df1 = pipeline.data_cleaning(test_raw)        
        print("Aqui1")
        print(df1.head())

        
        #feature engineering
        df2 = pipeline.feature_engineering(df1)
        test_return = df2.copy()
        print("Aqui2")
        print(df2.head())

        
        #data preparation        
        df3 = pipeline.data_preparation(df2)
        print("Aqui3")
        print(df3.head())

        
        #prediction
        df_response = pipeline.get_prediction(model, test_return, df3)
        print("Aqui4")
        
        
        
        return df_response
                     
    else:
        return Response( "{}", status = 200, mimetype = "application/json")
    
if __name__ == "__main__":
    port = os.environ.get('PORT', 5000)
    app.run(host = "0.0.0.0", debug = True)
