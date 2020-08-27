# -*- coding: utf-8 -*-
# import importlib
import importlib.util
import os
# from ctypes import CDLL, RTLD_GLOBAL


def init_darknet_lib():
    # return CDLL(os.path.join(
    #         os.environ.get("DARKNET_PATH", "./"),
    #         "libdarknet.so"), RTLD_GLOBAL)
    # MODULE_PATH = "/path/to/your/module/__init__.py"
    MODULE_NAME = "darknet"
    # import importlib
    # import sys
    # spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    #
    module_path = os.getenv("DARKNET_LIB_PYTHON_PATH", None)
    if not module_path:
        raise EnvironmentError("Path to darknet.py not set!")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    importlib.util.module_from_spec(spec)
    module = importlib.util.module_from_spec(spec)
    return spec.loader.exec_module(module)


darknet = init_darknet_lib()

