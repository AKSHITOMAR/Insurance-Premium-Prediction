# step-1:- use python 3.11 base image
FROM python:3.11-slim

# step-2 :- set working directory
WORKDIR /app

# step-3 :- copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of application code
COPY . .

# step-5:- Expose the port
EXPOSE 8000

# step-6:- Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]