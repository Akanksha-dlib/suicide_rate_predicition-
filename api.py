from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

class InputData(BaseModel):
    State: str
    Number_of_Suicides: int
    Percentage_share_in_total_suicides: float
    Projected_mid_year_population_in_lakhs: float
    Total_2020: int
    Percentage_variation: float
    Male: int
    Female: int
    Transgender: int
    Year: int

@app.post("/predict")
async def predict(data: InputData):
    # Load model and label encoder
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Prepare input data
    try:
        state_encoded = le.transform([data.State])[0]
    except ValueError:
        return {"error": "Invalid State value"}
    
    input_data = np.array([[
        state_encoded,
        data.Number_of_Suicides,
        data.Percentage_share_in_total_suicides,
        data.Projected_mid_year_population_in_lakhs,
        data.Total_2020,
        data.Percentage_variation,
        data.Male,
        data.Female,
        data.Transgender,
        data.Year
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return {"predicted_suicide_rate": float(prediction)}