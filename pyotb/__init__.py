# -*- coding: utf-8 -*-
"""This module provides convenient python wrapping of otbApplications."""
__version__ = "2.0.0.dev4"

from .helpers import logger, set_logger_level
from .apps import *
from .core import App, Input, Output, get_nbchannels, get_pixel_type, summarize, OTBObject

from .functions import (  # pylint: disable=redefined-builtin
    all,
    any,
    clip,
    define_processing_area,
    run_tf_function,
    where,
)
