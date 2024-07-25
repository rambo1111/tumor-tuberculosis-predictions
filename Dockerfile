FROM python:3.11-slim

# Install git and git lfs
RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install

# Set the working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/rambo1111/tumor-tuberculosis-predictions.git .
RUN git lfs pull

# Install dependencies
RUN pip install -r requirements.txt

# Command to run the application
CMD ["gunicorn", "app:app"]
