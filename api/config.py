"""API configuration via environment variables using pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Model paths
    MODEL_PATH: str = "model_artifacts/mobilenetv2_best.tflite"
    MODEL_META_PATH: str = "model_artifacts/mobilenetv2_meta.json"

    # Security
    MAX_FILE_SIZE_MB: int = 10
    RATE_LIMIT_PER_MINUTE: int = 20
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    API_KEY: str | None = None

    # Runtime
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


settings = Settings()
