from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pickle
import pandas as pd

app = FastAPI()

pickle_in = open("final_model.pkl","rb")
print(pickle_in)
classifier=pickle.load(pickle_in)


@app.get("/")
async def health_route(req: Request):
    """
    Health Route : Returns App details.

    """
    return JSONResponse(
        {
            "message": "hello world"
        }
    )

@app.get("/health")
async def health_route(req: Request):
    """
    Health Route : Returns App details.

    """
    return JSONResponse(
        {
            "message": "hello world health"
        }
    )


@app.get("/predict")
def health_route(req: Request):
    """
    Health Route : Returns App details.

    """
    dummy = pd.DataFrame([[50, 60, 70, 80, 90]], columns=['FirstScore', 'SecondScore', 'ThirdScore', 'FourthScore', 'FifthScore'])

    prediction = classifier.predict(dummy)
    print(prediction)
    return JSONResponse(
        {
            "message": prediction[0]
        }
    )


