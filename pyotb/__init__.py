# -*- coding: utf-8 -*-
"""This module provides convenient python wrapping of otbApplications."""
__version__ = "1.6.0"

from .helpers import find_otb, logger, set_logger_level

otb = find_otb()

from .apps import *

from .core import (
    App,
    Input,
    Output,
    get_nbchannels,
    get_pixel_type
)

from .functions import (  # pylint: disable=redefined-builtin
    all,
    any,
    where,
    clip,
    run_tf_function,
    define_processing_area
)
