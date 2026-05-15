__all__ = ["FlashOmniMethod"]


def __getattr__(name):
    if name == "FlashOmniMethod":
        from .method import FlashOmniMethod
        return FlashOmniMethod
    raise AttributeError(name)
