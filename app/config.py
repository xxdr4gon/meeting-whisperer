from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
	app_port: int = Field(default=8080, alias="APP_PORT")
	app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
	env: str = Field(default="production", alias="ENV")
	log_level: str = Field(default="info", alias="LOG_LEVEL")

	upload_dir: str = Field(default="/data/uploads", alias="UPLOAD_DIR")
	transcript_dir: str = Field(default="/data/transcripts", alias="TRANSCRIPT_DIR")
	models_dir: str = Field(default="/models", alias="MODELS_DIR")

	# Local admin fallback
	local_admin_enabled: bool = Field(default=True, alias="LOCAL_ADMIN_ENABLED")
	local_admin_username: str = Field(default="admin", alias="LOCAL_ADMIN_USERNAME")
	local_admin_password: str = Field(default="changeme", alias="LOCAL_ADMIN_PASSWORD")
	local_admin_email: str = Field(default="admin@example.com", alias="LOCAL_ADMIN_EMAIL")

	# IAM
	oidc_discovery_url: str | None = Field(default=None, alias="OIDC_DISCOVERY_URL")
	oidc_audience: str | None = Field(default=None, alias="OIDC_AUDIENCE")
	oidc_issuer: str | None = Field(default=None, alias="OIDC_ISSUER")
	oidc_client_id: str | None = Field(default=None, alias="OIDC_CLIENT_ID")
	oidc_client_secret: str | None = Field(default=None, alias="OIDC_CLIENT_SECRET")
	oidc_scopes: str = Field(default="openid profile email", alias="OIDC_SCOPES")
	oidc_redirect_uri: str | None = Field(default=None, alias="OIDC_REDIRECT_URI")
	oidc_post_logout_redirect_uri: str | None = Field(default=None, alias="OIDC_POST_LOGOUT_REDIRECT_URI")
	oidc_jwks_cache_ttl: int = Field(default=3600, alias="OIDC_JWKS_CACHE_TTL")
	oidc_allowed_algs: str = Field(default="RS256", alias="OIDC_ALLOWED_ALGS")

	ldap_url: str | None = Field(default=None, alias="LDAP_URL")
	ldap_bind_dn: str | None = Field(default=None, alias="LDAP_BIND_DN")
	ldap_bind_password: str | None = Field(default=None, alias="LDAP_BIND_PASSWORD")
	ldap_base_dn: str | None = Field(default=None, alias="LDAP_BASE_DN")
	ldap_user_attr: str | None = Field(default=None, alias="LDAP_USER_ATTR")
	ldap_group_attr: str | None = Field(default=None, alias="LDAP_GROUP_ATTR")

	saml_metadata_url: str | None = Field(default=None, alias="SAML_METADATA_URL")
	saml_sp_entity_id: str | None = Field(default=None, alias="SAML_SP_ENTITY_ID")
	saml_sp_acs_url: str | None = Field(default=None, alias="SAML_SP_ACS_URL")
	saml_sp_cert: str | None = Field(default=None, alias="SAML_SP_CERT")
	saml_sp_key: str | None = Field(default=None, alias="SAML_SP_KEY")
	saml_nameid_format: str | None = Field(default=None, alias="SAML_NAMEID_FORMAT")

	# SMTP
	smtp_host: str | None = Field(default=None, alias="SMTP_HOST")
	smtp_port: int = Field(default=587, alias="SMTP_PORT")
	smtp_user: str | None = Field(default=None, alias="SMTP_USER")
	smtp_password: str | None = Field(default=None, alias="SMTP_PASSWORD")
	smtp_tls: bool = Field(default=True, alias="SMTP_TLS")
	smtp_from: str = Field(default="noreply@example.com", alias="SMTP_FROM")

	# AI / Model
	model_type: str = Field(default="whisper", alias="MODEL_TYPE")
	model_path: str = Field(default="/models/ggml-base.bin", alias="MODEL_PATH")
	enable_gpu: bool = Field(default=False, alias="ENABLE_GPU")
	gpu_backend: str | None = Field(default=None, alias="GPU_BACKEND")
	gpu_device_id: int = Field(default=0, alias="GPU_DEVICE_ID")
	
	# Estonian ASR models from TalTechNLP
	use_estonian_model: bool = Field(default=True, alias="USE_ESTONIAN_MODEL")
	estonian_asr_dir: str = Field(default="/models/estonian-asr", alias="ESTONIAN_ASR_DIR")
	
	# English ASR models (faster-whisper)
	english_asr_dir: str = Field(default="/models/english-asr", alias="ENGLISH_ASR_DIR")
	
	# AI Summarization models
	use_ai_summarization: bool = Field(default=True, alias="USE_AI_SUMMARIZATION")  # Disabled by default due to model size issues
	ai_summarization_model: str = Field(default="qwen", alias="AI_SUMMARIZATION_MODEL")  # qwen (3B) or llama (8B)
	summarization_dir: str = Field(default="/models/summarization", alias="SUMMARIZATION_DIR")
	
	# Grammar Correction
	use_grammar_correction: bool = Field(default=True, alias="USE_GRAMMAR_CORRECTION")
	grammar_correction_dir: str = Field(default="/models/grammar-correction", alias="GRAMMAR_CORRECTION_DIR")

	# Pyannote diarization
	pyannote_auth_token: str | None = Field(default=None, alias="PYANNOTE_AUTH_TOKEN")
	pyannote_diarization_model: str = Field(default="pyannote/speaker-diarization-3.1", alias="PYANNOTE_DIARIZATION_MODEL")
	pyannote_local_path: str | None = Field(default=None, alias="PYANNOTE_LOCAL_PATH")

	# Security / CORS
	allowed_origins: List[str] = Field(default_factory=lambda: ["*"], alias="ALLOWED_ORIGINS")
	secret_key: str = Field(default="dev-secret-change", alias="SECRET_KEY")

	# Redis / Celery
	redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
	broker_url: str = Field(default="redis://redis:6379/1", alias="BROKER_URL")
	result_backend: str = Field(default="redis://redis:6379/2", alias="RESULT_BACKEND")

	class Config:
		env_file = ".env"
		case_sensitive = False
		protected_namespaces = ("settings_",)


settings = Settings()  # type: ignore
