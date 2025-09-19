from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from src.pipeline2.predict_pipeline import CustomData, PredictPipeline

# Initialize the FastAPI app
app = FastAPI()

# Set up the templates directory
templates = Jinja2Templates(directory="templates")

## Route for the home page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Render the home.html template when the user visits the root URL
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predictdata", response_class=HTMLResponse)
def predict_datapoint(
    request: Request,
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    # Create a CustomData object with the form data
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    
    # Convert the input data to a DataFrame
    pred_df = data.get_data_as_data_frame()
    
    # Use the prediction pipeline to get the result
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    # Render the result on the home.html page
    return templates.TemplateResponse("home.html", {"request": request, "results": round(results[0], 2)})

# This allows running the app directly with `python app.py` for simple testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)