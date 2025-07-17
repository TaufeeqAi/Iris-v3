FROM python:3.12-slim-bookworm

WORKDIR /app/frontend

# Copy requirements and install
COPY requirements.txt .          
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY . . 

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
