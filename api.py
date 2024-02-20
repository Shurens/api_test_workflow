# Import des librairies uvicorn, pickle, FastAPI, File, UploadFile, BaseModel
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np 
from pydantic import BaseModel
import pickle
import pandas as pd

import mlflow
import os
import boto3
import joblib

model_1 = joblib.load('model_1.pkl')
model_2 = joblib.load('model_2.pkl')


# Création des tags
tags = [
       {
              "name": "Hello",
              "description": "Retourne un Hello world basique",
       },
       {
              "name": "Predict V1",
              "description": "Pour les prédictions effectuées avec le modèle 1 sur Sleep Disorder",
       },
       {
              "name": "Predict V2",
              "description": "Pour les prédictions effectuées avec le modèle 2 sur Sleep Disorder"
       }
]

# Création de l'application
app = FastAPI(
       title="API de prediction",
       description= "API servant à faire des prédictions sur les catégories de Sleep Disorder d'une personne en fonction de certains paramètres",
       version= "1.0.0",
       openapi_tags= tags
)

# Point de terminaison avec paramètre
@app.get("/hello", tags=["Hello"])
def hello(name: str='World'):
       return {"message": f"Hello {name}"}



# Création du modèle de données pour le modéle 1 ('Gender', 'Age', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'BloodPressure_high', 'BloodPressure_low', 'Sleep Disorder'])
class Credit(BaseModel):
       Gender : int
       Age : int
       Physical_Activity_Level : int
       Heart_Rate : int
       Daily_Steps : int
       BloodPressure_high : int
       BloodPressure_low : int
       Sleep_Disorder : int


# Point de terminaison : Prédiction 1
@app.post("/predict", tags=["Predict V1"])
def predict_1(credit: Credit) :
       """
       Make a prediction on sleep disorder
       - **Gender**: Gender of the person 
       - **Age**: Age of the person
       - **Physical Activity Level**: Physical Activity Level of the person
       - **Heart Rate**: Heart Rate of the person
       - **Daily Steps**: Daily Steps of the person
       - **BloodPressure_high**: BloodPressure_high of the person
       - **BloodPressure_low**: BloodPressure_high of the person
       """
       data = dict(credit)
       data = {"Gender" : data["Gender"],
               "Age": data["Age"],
               "Physical Activity Level": data["Physical_Activity_Level"],
               "Heart Rate": data["Heart_Rate"],
               "Daily Steps": data["Daily_Steps"],
               "BloodPressure_high": data["BloodPressure_high"],
               "BloodPressure_low":data["BloodPressure_low"],
               }

       dataframe = pd.DataFrame([data])
       prediction = model_1.predict(dataframe)
       
                
       prediction = prediction[0].item()
       # with open('prediction.txt', 'a') as file:
       #        file.write(str(prediction) + '\n')
       return {"prediction" : prediction}


# Création du modèle de données pour le modéle 2 ('Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Sleep Disorder')
class Physique(BaseModel):
       Physical_Activity_Level : int
       Heart_Rate : int
       Daily_Steps : int
       Sleep_Disorder : int

# Point de terminaison : Prédiction 2
@app.post("/predict_2", tags=["Predict V2"])
def predict_2(physique: Physique) :
       """
       Make a prediction on sleep disorder with different parameters
       - **Physical Activity Level**: Physical Activity Level of the person
       - **Heart Rate**: Heart Rate of the person
       - **Daily Steps**: Daily Steps of the person
       """
       data = dict(physique)
       
       data = {"Physical Activity Level" : data["Physical_Activity_Level"],
              "Heart Rate" : data["Heart_Rate"],
              "Daily Steps" : data["Daily_Steps"]
       }
       df = pd.DataFrame([data])
       prediction = model_2.predict(df)
       prediction = prediction[0].item()
       return {"prediction" : prediction}
       
                
# Démarage de l'application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)