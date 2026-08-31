from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    catalog_token_admin: SecretStr = Field(min_length=1)

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        frozen=True,
        extra="ignore",
    )
