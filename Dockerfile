FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add debugging tools
RUN pip install ipython

# Create a directory for logs
RUN mkdir -p /app/logs

# Copy model and application files
COPY final.pt .
COPY app.py .

# Verify model file exists
RUN ls -la && \
    if [ -f "final.pt" ]; then \
        echo "Model found: $(ls -lh final.pt)"; \
    else \
        echo "WARNING: Model file NOT found!"; \
    fi

# Create a test script that exactly mirrors how inference.py loads the model
RUN echo 'import torch\n\
import sys\n\
\n\
# Set the path to your model\n\
MODEL_PATH = "final.pt"\n\
\n\
# Try the exact same loading approach as in inference.py\n\
print(f"Loading YOLO model from {MODEL_PATH}...")\n\
try:\n\
    # Try loading with Ultralytics YOLO\n\
    from ultralytics import YOLO\n\
    model = YOLO(MODEL_PATH)\n\
    print("SUCCESS: Model loaded with Ultralytics YOLO")\n\
except Exception as e:\n\
    print(f"Failed with Ultralytics: {e}")\n\
    try:\n\
        # Fallback to PyTorch Hub (YOLOv5)\n\
        model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)\n\
        print("SUCCESS: Model loaded with PyTorch Hub")\n\
    except Exception as e:\n\
        print(f"Failed with PyTorch Hub: {e}")\n\
        sys.exit(1)\n\
\n\
print("Model loaded successfully!")\n\
if hasattr(model, "names"):\n\
    print(f"Classes: {model.names}")\n\
' > test_model_loading.py

# Run the model loading test
RUN python test_model_loading.py || echo "WARNING: Model test failed but continuing build..."

# Expose port
EXPOSE 8000

# Set environment variables to help with loading
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

# Run the application with output logged
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 2>&1 | tee /app/logs/app.log"] 
