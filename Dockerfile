# Use the official Python 3.11 image from Docker Hub as the base image
FROM python:3.11.7

# Set the working directory inside the container
WORKDIR /app

# Install required dependencies, including sqlite3
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    && apt-get clean

RUN sqlite3 --version  # To verify the correct version

# Copy the requirements.txt file into the container
COPY requirements.txt ./requirements.txt

# Install the required dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Verify the path and installed packages (optional for debugging)
RUN echo $PATH
RUN which streamlit  # Verifies that streamlit is installed and in the path

# Expose the port that Streamlit will use
EXPOSE 8501

# Copy the entire application code into the container
COPY . /app

# Set the command to run the Streamlit app when the container starts
ENTRYPOINT ["streamlit", "run", "chatbot.py"]
