__all__ = ["SVG2Method"]


def __getattr__(name):
    if name == "SVG2Method":
        from .method import SVG2Method
        return SVG2Method
    raise AttributeError(name)
