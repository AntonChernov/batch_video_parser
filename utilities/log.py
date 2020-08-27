# -*- coding: utf-8 -*-
import logging
import os
import typing as typ
import pathlib as pth


def chek_or_create_path(file: typ.Union[str, pth.Path]) -> typ.Union[pth.Path, str]:
    f = file if isinstance(file, pth.Path) else pth.Path(file)
    dirs_pth = str(f).split(f.name)[0]
    if not os.path.exists(dirs_pth):
        os.makedirs(dirs_pth)
    return f


def logger_init(file: typ.Union[str, pth.Path] = None) -> logging.Logger:
    log_level = logging.getLevelName(os.environ.get("LOGGING_LEVEL"))
    logger = logging.getLogger("video_parser")
    logger.setLevel(log_level)
    # create file handler which logs even debug messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if file:
        f = chek_or_create_path(file)
        fh = logging.FileHandler(f)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        # create console handler with log level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        # create formatter and add it to the handlers
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
    return logger
