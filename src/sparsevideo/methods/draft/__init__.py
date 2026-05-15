__all__ = ["DraftMethod"]


def __getattr__(name):
    if name == "DraftMethod":
        from .method import DraftMethod
        return DraftMethod
    raise AttributeError(name)
