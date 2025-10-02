from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    EMBEDDING_PROVIDER: str = "local"  # local | openai
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER: str = "none"         # none | openai
    OPENAI_API_KEY: str = ""
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 6000

    # Pydantic v2 style
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )
