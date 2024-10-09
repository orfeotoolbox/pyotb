# -*- coding: utf-8 -*-
"""This module provides convenient python wrapping of otbApplications."""
__version__ = "2.1.0"

from .install import install_otb
from .helpers import logger
from .core import (
    OTBObject,
    App,
    Input,
    Output,
    get_nbchannels,
    get_pixel_type,
    summarize,
)
from .apps import *

from .functions import (  # pylint: disable=redefined-builtin
    all,
    any,
    clip,
    define_processing_area,
    run_tf_function,
    where,
)
