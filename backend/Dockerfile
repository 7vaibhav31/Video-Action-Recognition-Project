# Base image with Python 3.10
FROM python:3.10-slim

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER user

# Set the working directory
WORKDIR /home/user/app

# Copy the requirements file first for better caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Expose port 7860 as expected by Hugging Face Spaces
EXPOSE 7860

# Run the application using Gunicorn with a long timeout for cold starts
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "180", "--workers", "1", "--threads", "4", "app:app"]
