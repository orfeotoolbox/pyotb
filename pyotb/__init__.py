# -*- coding: utf-8 -*-
"""
This module provides convenient python wrapping of otbApplications
"""
__version__ = "1.3.3"

from .apps import *
from .core import App, Output, Input, get_nbchannels, get_pixel_type
from .functions import *  # pylint: disable=redefined-builtin
from .tools import logger
