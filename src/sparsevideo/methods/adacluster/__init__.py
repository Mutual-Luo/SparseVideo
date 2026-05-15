__all__ = ["AdaClusterMethod"]


def __getattr__(name):
    if name == "AdaClusterMethod":
        from .method import AdaClusterMethod
        return AdaClusterMethod
    raise AttributeError(name)
