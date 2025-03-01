from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from starlette.responses import RedirectResponse, Response
from hate.pipeline.train_pipeline import TrainPipeline
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import APP_HOST, APP_PORT  # Ensure these constants are defined appropriately
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for all origins (you can restrict this to your frontend domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a global instance of PredictionPipeline to load the model only once
prediction_pipeline = PredictionPipeline()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Define a Pydantic model for prediction requests
class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_route(request: PredictionRequest):
    try:
        # Extract text from the request body
        input_text = request.text
        # Run the prediction pipeline using the global instance
        result = prediction_pipeline.run_pipeline(input_text)
        return {"prediction": result}
    except Exception as e:
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
