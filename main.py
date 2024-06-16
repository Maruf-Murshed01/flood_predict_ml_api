#Bring in lightweight dependencies

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
    LATITUDE: float#22.7 #,
    LONGITUDE: float#90.36#,
    ALT: int #4#,
    Month: int #6#,
    Max_Temp: float #34.4#,
    Min_Temp: float#25.7#,
    Rainfall: int#512#,
    Relative_Humidity: int #80#,
    Wind_Speed: float #1.631481481#,
    Cloud_Coverage: float #5.6#,
    Bright_Sunshine: float#4.072340426#

with open('rf_model_ak.pkl', 'rb') as f:
    model = pickle.load(f)
     

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": float(yhat)}