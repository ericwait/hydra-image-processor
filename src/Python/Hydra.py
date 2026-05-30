"""Backwards-compatibility shim exposing the compiled extension as ``Hydra``.

The compiled CUDA extension is installed as
:mod:`hydra_image_processor.Hydra`. This shim re-exports it at the top level so
that ``import Hydra`` continues to resolve to the extension module.

New code should prefer the Pythonic wrapper API::

    import hydra_image_processor as HIP
"""

import sys as _sys

from hydra_image_processor import Hydra as _Hydra

_sys.modules[__name__] = _Hydra
