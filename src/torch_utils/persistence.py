# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Facilities for pickling Python code alongside other data.

The pickled code is automatically imported into a separate Python module
during unpickling. This way, any previously exported pickles will remain
usable even if the original code is no longer available, or if the current
version of the code is not consistent with what was originally pickled."""

import sys
import pickle
import io
import inspect
import copy
import uuid
import types
import dnnlib

# Internal versioning and tracking variables
_version = 6  # Internal version number
_decorators = set()  # Set of decorator classes
_import_hooks = []  # List of import hook functions
_module_to_src_dict = dict()  # Mapping of modules to source code
_src_to_module_dict = dict()  # Mapping of source code to modules

def persistent_class(orig_class):
    """
    Class decorator to make a Python class persistent by saving its source code when pickled.

    Args:
        orig_class (type): The original class to be decorated.

    Returns:
        type: Decorated class with persistence capabilities.
    """
    assert isinstance(orig_class, type)

    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_args = copy.deepcopy(args)
            self._init_kwargs = copy.deepcopy(kwargs)
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            """Get the initialization arguments of the instance."""
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            """Get the initialization keyword arguments of the instance."""
            return dnnlib.EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            """Customize object reduction for pickling."""
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = {
                    'type': 'class',
                    'version': _version,
                    'module_src': self._orig_module_src,
                    'class_name': self._orig_class_name,
                    'state': fields[2]
                }
                fields[0] = _reconstruct_persistent_obj  # Reconstruction function
                fields[1] = (meta,)  # Reconstruction arguments
                fields[2] = None  # State dictionary
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    _decorators.add(Decorator)
    return Decorator

def is_persistent(obj):
    """
    Check if an object or class is persistent.

    Args:
        obj: Object or class to check.

    Returns:
        bool: True if the object or class is persistent, False otherwise.
    """
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators