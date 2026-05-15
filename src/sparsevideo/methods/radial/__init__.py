__all__ = ["RadialMethod"]


def __getattr__(name):
    if name == "RadialMethod":
        from .method import RadialMethod
        return RadialMethod
    raise AttributeError(name)
