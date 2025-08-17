# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application's code into the container at /app
COPY . .

# Define the command to run the application
# The command can be overridden when running the container
# For example, to process a specific resume file
ENTRYPOINT ["python", "main.py"]

# Default command if no arguments are provided to `docker run`
CMD ["--help"]
