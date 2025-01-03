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