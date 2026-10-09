from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REFUSAL_MESSAGE = "I couldn't find this information in the college knowledge base."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    llm_max_retries: int = 4

    top_k: int = 8
    # Measured on this embedding model: correct matches score 0.18-0.63, off-topic questions 0.74-0.89.
    max_distance: float = 0.65

    data_dir: Path = Path("./data")

    database_url: str = ""

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'college.db').as_posix()}"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
