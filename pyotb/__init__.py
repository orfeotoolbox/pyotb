# -*- coding: utf-8 -*-
"""This module provides convenient python wrapping of otbApplications."""
__version__ = "2.0.0"

from .helpers import logger, set_logger_level
from .apps import *
from .core import (
    App,
    Input,
    Output,
    get_nbchannels,
    get_pixel_type,
    summarize
)
from .functions import (  # pylint: disable=redefined-builtin
    all,
    any,
    where,
    clip,
    run_tf_function,
    define_processing_area
)
