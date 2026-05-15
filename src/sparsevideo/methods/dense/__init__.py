__all__ = ["DenseMethod"]


def __getattr__(name):
    if name == "DenseMethod":
        from .method import DenseMethod
        return DenseMethod
    raise AttributeError(name)
