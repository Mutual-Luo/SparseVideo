__all__ = ["SVG1Method"]


def __getattr__(name):
    if name == "SVG1Method":
        from .method import SVG1Method
        return SVG1Method
    raise AttributeError(name)
