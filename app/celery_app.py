from celery import Celery
from .config import settings

celery_app = Celery(
    "transcriber",
    broker=settings.broker_url,
    backend=settings.result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
)
