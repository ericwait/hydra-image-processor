"""Backwards-compatibility shim for the legacy top-level ``HIP`` module.

Historically the conda-forge package shipped a top-level ``HIP`` C extension
exposing the raw Hydra API (e.g. ``HIP.Gaussian``). That extension is now
installed as :mod:`hydra_image_processor.Hydra`. This shim re-exports it so
that existing ``import HIP`` code keeps working unchanged.

New code should prefer the Pythonic wrapper API::

    import hydra_image_processor as HIP

    HIP.gaussian(image, sigmas=[2, 2, 1])
"""

import sys as _sys

from hydra_image_processor import Hydra as _Hydra

# Make ``import HIP`` resolve to the compiled extension module itself, matching
# the historical top-level module one-to-one.
_sys.modules[__name__] = _Hydra
