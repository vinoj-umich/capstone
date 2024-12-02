# Use the official Python 3.11 image from Docker Hub as the base image
FROM python:3.11.7

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt ./requirements.txt

# Install the required dependencies from the requirements.txt file
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Expose the port that Streamlit will use
EXPOSE 8501

# Copy the entire application code into the container
COPY . /app

# Set the command to run the Streamlit app when the container starts
ENTRYPOINT ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
