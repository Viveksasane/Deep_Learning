FROM python:3.10-slim


# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    build-essential \
    awscli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app



# 
COPY . /app


# Copy only requirements first (Docker layer caching)
COPY requirements.txt .


# Upgrade pip & install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt






# Default command
CMD ["python", "app.py"]




