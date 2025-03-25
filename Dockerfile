FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
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

# Verify model file exists and check ultralytics version
RUN ls -la && \
    if [ -f "final.pt" ]; then \
        echo "Model found: $(ls -lh final.pt)"; \
    else \
        echo "WARNING: Model file NOT found!"; \
    fi && \
    python -c "from ultralytics import __version__ as version; print(f'Ultralytics version: {version}')"

# Test model loading with ultralytics (the method that worked in our tests)
RUN echo 'from ultralytics import YOLO; \
print("Testing model loading with Ultralytics..."); \
try: \
    model = YOLO("final.pt"); \
    print("SUCCESS: Model loaded with Ultralytics!"); \
    print(f"Model classes: {model.names}"); \
except Exception as e: \
    print(f"ERROR: {e}"); \
' > test_ultralytics.py

# Try to load the model with our test script but don't fail if it doesn't work
RUN python test_ultralytics.py || echo "Model test failed but continuing build..."

# Expose port
EXPOSE 8000

# Set environment variable to enable better error messages
ENV PYTHONFAULTHANDLER=1

# Run the application with output logged
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 2>&1 | tee /app/logs/app.log"] 
