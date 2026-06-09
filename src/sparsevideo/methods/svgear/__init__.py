__all__ = ["SVGEARMethod"]


def __getattr__(name):
    if name == "SVGEARMethod":
        from .method import SVGEARMethod
        return SVGEARMethod
    raise AttributeError(name)
