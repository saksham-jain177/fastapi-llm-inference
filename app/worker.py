import os
from celery import Celery

# Configure Celery to use Redis as Broker
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")

celery_app = Celery(
    "worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_BROKER_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(name="train_model_task")
def train_model_task():
    """
    Background Task: Runs the training script.
    """
    import subprocess
    # Run the script we moved to scripts/
    subprocess.run(["python", "scripts/train_rlhf.py"], check=True)
    return "Training Completed"
