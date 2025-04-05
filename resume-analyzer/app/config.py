"""
Handles application configuration using Pydantic's BaseSettings.

Reads settings from environment variables and/or a .env file,
providing a centralized configuration object for the application.
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Explicitly load environment variables from a .env file into the environment.
# Although BaseSettings can load from .env directly via its Config class,
# calling load_dotenv() here ensures variables are loaded early and available
# via os.getenv immediately if needed elsewhere. It's generally safe
# and common practice.
load_dotenv()

class Settings(BaseSettings):
    """
    Defines application settings, loading values from environment variables or .env file.

    Attributes:
        google_api_key (str): The API key for Google Generative AI services.
                               Loaded from the 'GOOGLE_API_KEY' environment variable or .env.
        llm_model_name (str): The specific Gemini model to use for chat/language generation.
        embedding_model_name (str): The specific Google model to use for text embeddings.
    """
    # --- Google AI Configuration ---
    # Reads 'GOOGLE_API_KEY' from environment or .env file.
    # Falls back to 'MISSING_API_KEY' if not found by BaseSettings.
    # Type hint 'str' ensures Pydantic expects a string value.
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "MISSING_API_KEY")

    # --- Model Configuration ---
    # Default model names - can be overridden by environment variables
    # (e.g., setting 'LLM_MODEL_NAME' in .env or environment).
    llm_model_name: str = "gemini-1.5-flash-latest"
    embedding_model_name: str = "models/embedding-001"

    # --- Pydantic BaseSettings Configuration ---
    class Config:
        """
        Inner class to configure the behavior of the BaseSettings model.
        """
        # Specifies the name of the environment file to load variables from.
        env_file: str = '.env'
        # Allows loading variables even if the .env file is missing.
        # Set to '.env' to enforce its presence if needed.
        # env_file_encoding = 'utf-8' # Optional: Specify encoding

        # Ignores any extra variables found in the environment or .env file
        # that do not correspond to fields defined in this Settings class.
        # Set to 'forbid' to raise an error on extra fields.
        extra: str = 'ignore'


# Create a single instance of the Settings class.
# This instance will be imported and used throughout the application
# to access configuration values in a consistent way.
settings: Settings = Settings()

# --- Runtime Check for Critical Settings ---
# Perform a basic check after attempting to load settings to ensure
# critical variables like the API Key are present.
# This helps catch configuration errors early during application startup.
if settings.google_api_key == "MISSING_API_KEY" or not settings.google_api_key:
    # Print a clear warning to the console if the key seems missing.
    print("\n--- WARNING ---")
    print("Configuration Issue: GOOGLE_API_KEY not found in environment variables or the .env file.")
    print("The application requires this key to interact with Google AI services and may not function correctly.")
    print("Please ensure the key is set in your .env file or environment.")
    print("---------------\n")