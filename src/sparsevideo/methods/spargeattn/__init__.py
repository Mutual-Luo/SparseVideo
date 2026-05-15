__all__ = ["SpargeAttnMethod"]


def __getattr__(name):
    if name == "SpargeAttnMethod":
        from .method import SpargeAttnMethod
        return SpargeAttnMethod
    raise AttributeError(name)
