"""Trainer implementations."""



def __getattr__(name: str):
    """Preserve public trainer exports without importing torch for CLI preflight."""
    from importlib import import_module

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{name[:-7].lower()}_trainer", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "CPTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "GSPOTrainer",
    "SFTTrainer",
]
