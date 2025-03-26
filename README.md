# Car Damage Detection API

A FastAPI application for car damage detection using a YOLO-based model.

## Features

- Upload an image to detect car damage
- Returns a processed image with bounding boxes around detected damage
- Provides detailed information about detected objects including class, confidence, and coordinates
- Simple health check endpoint to verify API is operational

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/amangupta20/carddapi.git
   cd carddapi
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:

   ```
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### GET `/`

- Returns a welcome message to confirm the API is running

### GET `/health`

- Checks if the API is healthy and if the model is loaded properly

### POST `/detect`

- Endpoint for car damage detection
- Parameters:
  - `file`: Image file to be analyzed (multipart/form-data)
  - `confidence`: Confidence threshold (optional, default: 0.25)
- Returns:
  - `processed_image`: Base64 encoded image with bounding boxes
  - `inference_time`: Time taken for inference in seconds
  - `total_objects`: Total number of objects detected
  - `detections`: List of detection results with class, confidence, and coordinates
  - `class_counts`: Count of each class detected

## Example

Using curl:

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg" \
     -F "confidence=0.25"
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("path/to/your/image.jpg", "rb")}
data = {"confidence": 0.25}

response = requests.post(url, files=files, data=data)
result = response.json()

# The result contains:
# - processed_image: Base64 encoded image with bounding boxes
# - inference_time: Time taken for inference
# - total_objects: Number of objects detected
# - detections: List of detection results
# - class_counts: Count of each class detected
```

## Docker Support (Optional)

To run the API in a Docker container:

1. Build the Docker image:

   ```
   docker build -t car-damage-detection .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 car-damage-detection
   ```
