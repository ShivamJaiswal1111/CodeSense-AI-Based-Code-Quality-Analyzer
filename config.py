"""
CodeSense - Configuration Management
Loads and validates all configuration from environment variables and defaults.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    path: str          = os.getenv("DB_PATH", "codesense.db")
    pool_size: int     = int(os.getenv("DB_POOL_SIZE", "5"))
    timeout: int       = int(os.getenv("DB_TIMEOUT", "30"))
    echo: bool         = os.getenv("DB_ECHO", "false").lower() == "true"


@dataclass
class AuthConfig:
    secret_key: str    = os.getenv("SECRET_KEY", "change-me-in-production-32chars!!")
    otp_expiry: int    = int(os.getenv("OTP_EXPIRY_MINUTES", "10"))
    max_attempts: int  = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    session_hours: int = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
    bcrypt_rounds: int = int(os.getenv("BCRYPT_ROUNDS", "12"))
    smtp_host: str     = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int     = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str     = os.getenv("SMTP_USER", "")
    smtp_pass: str     = os.getenv("SMTP_PASS", "")
    from_email: str    = os.getenv("FROM_EMAIL", "noreply@codesense.ai")


@dataclass
class MLConfig:
    model_path: str    = os.getenv("MODEL_PATH", "models/codesense_model.pkl")
    scaler_path: str   = os.getenv("SCALER_PATH", "models/codesense_scaler.pkl")
    features_path: str = os.getenv("FEATURES_PATH", "models/feature_names.json")
    retrain_days: int  = int(os.getenv("MODEL_RETRAIN_DAYS", "7"))
    min_samples: int   = int(os.getenv("MIN_TRAIN_SAMPLES", "5000"))
    target_r2: float   = float(os.getenv("TARGET_R2", "0.90"))


@dataclass
class CacheConfig:
    ttl: int           = int(os.getenv("CACHE_TTL", "3600"))
    max_size: int      = int(os.getenv("CACHE_MAX_SIZE", "500"))
    directory: str     = os.getenv("CACHE_DIR", "cache")
    enabled: bool      = os.getenv("CACHE_ENABLED", "true").lower() == "true"


@dataclass
class AnalysisConfig:
    timeout: int       = int(os.getenv("ANALYSIS_TIMEOUT", "30"))
    max_file_mb: float = float(os.getenv("MAX_FILE_MB", "5"))
    max_lines: int     = int(os.getenv("MAX_CODE_LINES", "5000"))
    min_lines: int     = int(os.getenv("MIN_CODE_LINES", "3"))
    parallel: bool     = os.getenv("PARALLEL_ANALYSIS", "true").lower() == "true"
    max_workers: int   = int(os.getenv("MAX_WORKERS", "4"))


@dataclass
class AppConfig:
    debug: bool        = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str     = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str       = os.getenv("LOG_DIR", "logs")
    port: int          = int(os.getenv("PORT", "8501"))
    env: str           = os.getenv("ENVIRONMENT", "development")

    db: DatabaseConfig     = field(default_factory=DatabaseConfig)
    auth: AuthConfig       = field(default_factory=AuthConfig)
    ml: MLConfig           = field(default_factory=MLConfig)
    cache: CacheConfig     = field(default_factory=CacheConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def __post_init__(self) -> None:
        """Ensure required directories exist."""
        for directory in [self.log_dir, self.cache.directory,
                          Path(self.ml.model_path).parent]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def is_production(self) -> bool:
        return self.env == "production"

    def to_dict(self) -> dict:
        return {
            "debug": self.debug,
            "env": self.env,
            "log_level": self.log_level,
        }


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the singleton AppConfig, creating it on first call."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """Force reload configuration (useful after env changes)."""
    global _config
    load_dotenv(override=True)
    _config = AppConfig()
    return _config