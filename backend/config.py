from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REFUSAL_MESSAGE = "I couldn't find this information in the college knowledge base."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any OpenAI-compatible chat endpoint. Free options: Groq, Google Gemini, or a
    # local Ollama server. Defaults to Groq's free tier (needs a free API key).
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    llm_max_retries: int = 4  # SDK retries 429/5xx with backoff, honouring Retry-After

    # Retrieval over the policy/notice snippets
    top_k: int = 8
    # Max cosine distance (0 = identical). Farther snippets are discarded.
    # Measured with the bundled MiniLM model: correct matches score 0.18-0.63,
    # clearly off-topic questions 0.74-0.89, so 0.65 only removes the obvious misses.
    # Borderline cases (0.59-0.65) are decided by the answer step, not by distance.
    max_distance: float = 0.65

    data_dir: Path = Path("./data")

    # Empty = local SQLite file in data_dir. For Supabase, paste the Postgres
    # connection string (use the "postgresql+psycopg://" scheme).
    database_url: str = ""

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'college.db').as_posix()}"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
