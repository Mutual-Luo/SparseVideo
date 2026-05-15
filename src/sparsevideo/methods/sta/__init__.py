__all__ = ["STAMethod"]


def __getattr__(name):
    if name == "STAMethod":
        from .method import STAMethod
        return STAMethod
    raise AttributeError(name)
