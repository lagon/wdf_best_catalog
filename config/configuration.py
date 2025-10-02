from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Add your configuration fields here with type hints.
    """

    main_results_path_dir: str
    open_ai_api_key: str

    web_cache_dir: str
    product_summary_requests_cache_dir: str
    product_details_request_cache_dir: str
    product_grouping_request_cache_dir: str

    product_web_extraction_model_name: str #     model_name = "gpt-4.1-mini"
    product_summary_extraction_model_name: str #     model_name = "gpt-4.1-mini"
    hierarchy_inference_model_name: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )


@lru_cache
def get_settings() -> Settings:
    """
    Create and cache a single Settings instance.
    This ensures the .env file is only read once.
    """
    return Settings()

