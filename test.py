import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from api import app

@pytest.mark.asyncio
async def test_predict():
    # Define the input data
    input_data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Make a POST request to the /predict endpoint
        response = await client.post("/predict", json=input_data)
        
        # Assert the response status code
        assert response.status_code == 200
        
        # Assert the response JSON structure
        response_json = response.json()
        assert "prediction" in response_json
        assert isinstance(response_json["prediction"], int)
