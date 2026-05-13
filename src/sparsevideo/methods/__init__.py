from .._registry import register_method

from .dense import DenseMethod

register_method("dense", DenseMethod)

try:
    from .topk import TopKMethod
    register_method("topk", TopKMethod)
except ImportError:
    pass

try:
    from .spargeattn import SpargeAttnMethod
    register_method("spargeattn", SpargeAttnMethod)
except ImportError:
    pass

try:
    from .svg import SVGMethod
    register_method("svg", SVGMethod)
except ImportError:
    pass

try:
    from .sap import SAPMethod
    register_method("sap", SAPMethod)
except ImportError:
    pass

try:
    from .radial import RadialMethod
    register_method("radial", RadialMethod)
except ImportError:
    pass

try:
    from .sta import STAMethod
    register_method("sta", STAMethod)
except ImportError:
    pass

try:
    from .draft import DraftMethod
    register_method("draft", DraftMethod)
except ImportError:
    pass

try:
    from .adacluster import AdaClusterMethod
    register_method("adacluster", AdaClusterMethod)
except ImportError:
    pass

try:
    from .flashomni import FlashOmniMethod
    register_method("flashomni", FlashOmniMethod)
except ImportError:
    pass

try:
    from .svoo import SVOOMethod
    register_method("svoo", SVOOMethod)
except ImportError:
    pass
