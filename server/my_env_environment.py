"""The old name of server/inventory_environment.py, kept so that existing imports keep working."""

try:
    from inventory_environment import InventoryEnvironment, PharmaEnvironment
except ModuleNotFoundError:
    from server.inventory_environment import InventoryEnvironment, PharmaEnvironment

__all__ = ["InventoryEnvironment", "PharmaEnvironment"]
