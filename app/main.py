from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from pydantic import BaseModel
app = FastAPI()

pickle_in = open("final_model.pkl","rb")

classifier=pickle.load(pickle_in)


@app.get("/")
async def health_route(req: Request):
    """
    Health Route : Returns App details.

    """
    return JSONResponse(
        {
            "message": "welcome to prediction module of lpu smart prep"
        }
    )



class PredictionRequest(BaseModel):
    data: list


@app.post("/predict")
async def predict_model(data: PredictionRequest):
    """
    for prediction test result

    """
    print(data.data)
    
    dummy = pd.DataFrame([data.data], columns=['FirstScore', 'SecondScore', 'ThirdScore', 'FourthScore', 'FifthScore'])

    prediction = classifier.predict(dummy)
    
    return JSONResponse(
        {
            "message": prediction[0]
        }
    )


