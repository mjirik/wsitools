# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""

from loguru import logger
import os.path as op
import sys

from wsitools import annotation

# print("ann to json")

if __name__ == "__main__":
    # print(sys.argv)
    pth = op.expanduser(sys.argv[1])
    # print(pth)
    annotation.ndpa_to_json(pth)
