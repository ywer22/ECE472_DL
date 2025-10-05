from importlib.resources import files
from typing import List

from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""

    batch_size: int = 128
    val_split: float = 0.1


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 128
    num_iters: int = 500
    learning_rate: float = 0.001


class ModelSettings(BaseModel):
    """Settings for model configuration."""

    input_depth: int = 1
    layer_depths: List[int] = [32, 64]
    layer_kernel_sizes: List[List[int]] = [[3, 3], [3, 3]]  # cast to list of lists
    num_classes: int = 10
    dropout: float = 0.5
    l2_reg: float = 1e-4

    # validator to convert list of lists to list of tuples
    @field_validator("layer_kernel_sizes")
    @classmethod
    def convert_to_tuples(cls, v):
        """Convert list of lists to list of tuples for the model."""
        return [tuple(kernel) for kernel in v]


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw03").joinpath("config.toml"),
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
