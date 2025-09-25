from celery import Celery
from .config import settings

celery_app = Celery(
	"transcriber",
	broker=settings.broker_url,
	backend=settings.result_backend,
)

# Ensure tasks are discovered/registered
celery_app.conf.update(
	task_serializer="json",
	result_serializer="json",
	accept_content=["json"],
	task_track_started=True,
	worker_send_task_events=True,
	task_send_sent_event=True,
	imports=("app.tasks",),
)

# Try autodiscovery as well (harmless if already imported)
celery_app.autodiscover_tasks(["app"])  # looks for tasks.py inside the package

# Force-import to register tasks in environments that skip imports
try:
	from . import tasks  # noqa: F401
except Exception:
	pass
