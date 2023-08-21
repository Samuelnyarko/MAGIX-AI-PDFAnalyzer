
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the content to the docker container
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Command to run on container start
CMD ["streamlit", "run", "app.py"]

