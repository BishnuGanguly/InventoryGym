"""
InventoryGym: an inventory-management environment for training and testing LLM agents.

    from inventorygym import InventoryEnvClient, InventoryAction

The modules of this project import each other by their plain names (`models`, `tasks`,
`world`, ...), the layout OpenEnv expects for an environment. So that they are found
when the project is installed as the package `inventorygym`, this file puts the
package's own folder on the import path.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from client import InventoryEnvClient, PharmaEnvClient  # noqa: E402
from models import (  # noqa: E402
    EpisodeConfig,
    InventoryAction,
    InventoryState,
    PharmaAction,
    SKUEpisodeConfig,
    SKUState,
)

from version import __version__  # noqa: E402

__all__ = [
    "InventoryEnvClient",
    "InventoryAction",
    "InventoryState",
    "SKUState",
    "EpisodeConfig",
    "SKUEpisodeConfig",
    # Names from before the project became InventoryGym.
    "PharmaEnvClient",
    "PharmaAction",
    "__version__",
]
