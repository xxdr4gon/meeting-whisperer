import smtplib
from email.mime.text import MIMEText

from ..config import settings


def send_transcript_email(to_email: str, job_id: str, body_text: str) -> None:
	if not settings.smtp_host or not settings.smtp_user or not settings.smtp_password:
		return
	msg = MIMEText(body_text, _charset="utf-8")
	msg["Subject"] = f"Transcript for job {job_id}"
	msg["From"] = settings.smtp_from
	msg["To"] = to_email
	server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
	try:
		if settings.smtp_tls:
			server.starttls()
		server.login(settings.smtp_user, settings.smtp_password)
		server.sendmail(settings.smtp_from, [to_email], msg.as_string())
	finally:
		server.quit()
