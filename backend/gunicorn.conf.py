# Gunicorn configuration file for Render deployment

# Increase worker timeout to 180 seconds to allow TensorFlow/ResNet50 to load on cold start
timeout = 180

# Restrict the number of workers to 1 to stay within the 512 MB memory limit of Render's free tier
workers = 1
