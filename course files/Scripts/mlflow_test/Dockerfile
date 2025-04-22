FROM python:3.12.2-slim

# Install MLflow and dependencies
RUN pip install mlflow

# Expose the MLflow port
EXPOSE 5000

# Command to run when container starts
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]