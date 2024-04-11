#! /usr/bin/python
# -*- coding: utf-8 -*-


from loguru import logger

import zipfile
import os
import os.path as op
import tempfile
import sys
import glob

# add deprecation warning during import of this module
logger.warning("This module is deprecated. Use openslide and openslide_bin instead.")

OPENSLIDE_PATH = r"~\Downloads\openslide-win64-20171122\bin"
OPENSLIDE_URL = "https://github.com/openslide/openslide-winbuild/releases/download/v20171122/openslide-win64-20171122.zip"
# OPENSLIDE_URL = "https://github.com/openslide/openslide-bin/releases/download/v4.0.0.2/openslide-bin-4.0.0.2-windows-x64.zip"

if sys.version_info < (3, 0):
    import urllib as urllibr
else:
    import urllib.request as urllibr


def import_openslide(use_openslide_bin=True):
    if use_openslide_bin:
        try:
            import openslide_bin
            import openslide
        except ImportError as e:
            logger.debug("Import openslide_bin failed. Try to use libfixer.import_openslide(use_openslide_bin=False)")
            raise e
    else:
        import_openslide_without_openslide_bin()

def import_openslide_without_openslide_bin():
    if os.name == "nt":
        pth = op.expanduser(OPENSLIDE_PATH)
        dll_list = glob.glob(op.join(pth, "*.dll"))
        if len(dll_list) < 5:
            print("Trying to download openslide dll files in {}".format(pth))
            libfix()
        # pth = op.expanduser(r"~\projects\scaffan\devel\knihovny")
        # pth = op.expanduser(r"~\Miniconda3\envs\lisa36\Library\bin")
        sys.path.insert(0, pth)
        orig_PATH = os.environ["PATH"]
        orig_split = orig_PATH.split(";")
        if pth not in orig_split:
            logger.debug("add path {}".format(pth))
        os.environ["PATH"] = pth + ";" + os.environ["PATH"]



def download_and_unzip(url, outdir):
    # try:
    #     import pywget
    # except:
    #     import wget as pywget

    if outdir is None:

        outdir = tempfile.gettempdir()
        # print("temp directory ", outdir)
        outdir = tempfile.mkdtemp()
    # there is problem with wget. It uses its ouwn tempfile in current dir. It is not sure that there will
    # be requred permisson for write

    if not op.exists(outdir):
        os.makedirs(outdir)
    filename = op.join(outdir, "openslides.zip")
    urllibr.urlretrieve(url, filename)
    # filename = pywget.download(url, out=filename)

    zf = zipfile.ZipFile(filename)
    zf.extractall(outdir)
    zf.close()
    return outdir


def libfix():

    # glob.glob(op.expanduser("~/Downloads/"))
    if sys.platform.startswith("win"):
        # print("Trying to download .dll libraries")
        libfix_windows()
    if sys.platform.startswith("linux"):
        # print("Trying to download .so libraries")
        pass
        # libfix_linux_conda()


# def get_conda_dir():
#     """
#     Get conda base dir.
#     I would remove envs subdirs from output dir.
#
#     :return:
#     """
#     if sys.version_info.major == 2:
#         conda_dir = get_conda_dir_old()
#     else:
#         conda_dir = sys.exec_prefix
#     idx = conda_dir.find("envs")
#     if idx > 0:
#         conda_dir = conda_dir[:idx]
#     return conda_dir


def libfix_windows(url=None):
    if url is None:
        if sys.maxsize > 2 ** 32:
            url = "https://github.com/openslide/openslide-winbuild/releases/download/v20171122/openslide-win64-20171122.zip"
        else:
            url = "https://github.com/openslide/openslide-winbuild/releases/download/v20171122/openslide-win32-20171122.zip"
    outdir = download_and_unzip(url, op.expanduser("~/Downloads/"))

    # dest_dir = get_conda_dir()

    # for file in glob.glob(r'ITK+Skelet3D_dll/*.dll'):#
    # try:
    #     shutil.rmtree(outdir)
    # except:
    #     shutil.copy(file, dest_dir)
    #     print("copy %s ---> %s" % (file, dest_dir))

    #     import traceback
    #     traceback.print_exc()
