FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory and copy code
WORKDIR /app
COPY . .

CMD ["python", "Image2Video.py"]
