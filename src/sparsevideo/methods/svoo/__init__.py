__all__ = ["SVOOMethod"]


def __getattr__(name):
    if name == "SVOOMethod":
        from .method import SVOOMethod
        return SVOOMethod
    raise AttributeError(name)
