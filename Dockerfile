FROM python:3.8

WORKDIR /app

# Accept build arguments
ARG COMET_API_KEY
ARG COMET_PROJECT_NAME
ARG COMET_WORKSPACE

# Set environment variables
ENV COMET_API_KEY=${COMET_API_KEY}
ENV COMET_PROJECT_NAME=${COMET_PROJECT_NAME}
ENV COMET_WORKSPACE=${COMET_WORKSPACE}

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the model training script and run it to create the model
COPY simple_model.py .
RUN python simple_model.py

# Copy the FastAPI app
COPY api.py .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
