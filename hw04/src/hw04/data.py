from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class Data:
    """Handles loading and batching of the CIFAR-10 dataset."""
