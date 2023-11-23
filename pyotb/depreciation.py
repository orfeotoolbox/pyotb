"""Helps with deprecated classes and methods.

Taken from https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias
"""
from typing import Callable, Dict, Any
import functools
import warnings


WARN = "\033[91m"
ENDC = "\033[0m"
OKAY = "\033[92m"


def depreciation_warning(message: str):
    """Shows a warning message.

    Args:
        message: message to log

    """
    warnings.warn(
        message=message,
        category=DeprecationWarning,
        stacklevel=3,
    )


def deprecated_alias(**aliases: str) -> Callable:
    """Decorator for deprecated function and method arguments.

    Use as follows:

    @deprecated_alias(old_arg='new_arg')
    def myfunc(new_arg):
        ...

    Args:
        **aliases: aliases

    Returns:
        wrapped function

    """

    def deco(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(func_name: str, kwargs: Dict[str, Any], aliases: Dict[str, str]):
    """Helper function for deprecating function arguments.

    Args:
        func_name: function
        kwargs: keyword args
        aliases: aliases

    Raises:
        ValueError: if both old and new arguments are provided

    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise ValueError(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )
            message = (
                f"{WARN}`{alias}`{ENDC} is deprecated as an argument to "
                f"`{func_name}`; use {OKAY}`{new}`{ENDC} instead."
            )
            depreciation_warning(message)
            kwargs[new] = kwargs.pop(alias)


def deprecated_attr(replacement: str) -> Callable:
    """Decorator for deprecated attr.

    Use as follows:

    @deprecated_attr(replacement='new_attr')
    def old_attr(...):
        ...

    Args:
        replacement: name of the new attr (method or attribute)

    Returns:
        wrapped function

    """

    def deco(attr: Any):
        @functools.wraps(attr)
        def wrapper(self, *args, **kwargs):
            depreciation_warning(
                f"{WARN}`{attr.__name__}`{ENDC} will be removed in future "
                f"releases. Please replace {WARN}`{attr.__name__}`{ENDC} with "
                f"{OKAY}`{replacement}`{ENDC}."
            )
            out = getattr(self, replacement)
            return out(*args, **kwargs) if isinstance(out, Callable) else out

        return wrapper

    return deco
