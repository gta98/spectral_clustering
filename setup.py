from setuptools import setup, find_packages, Extension
import sysconfig
import distutils.command.build
import os

cwd = os.path.dirname(
        os.path.dirname(
            os.path.abspath(os.path.dirname(__file__)) ))
cwd = os.path.abspath(os.path.dirname(__file__))

PATH_SRC = os.environ.get('PATH_SRC') or f"{cwd}"
PATH_OUT = os.environ.get('PATH_OUT') or f"{cwd}"

PRECONFIGURED_CFLAGS = []#sysconfig.get_config_var('CFLAGS').split()

os.makedirs(f"{PATH_OUT}", exist_ok=True)

# Override build command
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = f"{PATH_OUT}"

# setup() parameters - https://packaging.python.org/guides/distributing-packages-using-setuptools/
setup(
    name='spkmeans',
    version='6.9.4.2.0',
    author="Spongebob Squarepants",
    author_email="a@xb.ax",
    description="This is an implementation of the spkmeans algorithm",
    install_requires=['invoke'],
    packages=find_packages(),
    ext_modules=[
        Extension(
            # the qualified name of the extension module to build
            'mykmeanssp',
            # the files to compile into our module relative to ``setup.py``
            [
                f"{PATH_SRC}/kmeans.c"
            ],
            extra_compile_args=[
                "-Wall", "-Wextra", "-Werror", "-lm",
                #"-pedantic-errors",
                "-Wno-error=missing-field-initializers", # FIXME - Issue in Python 3.8: https://github.com/SELinuxProject/setools/issues/31
                "-Wno-error=unused-parameter", # FIXME - what do I do with "PyObject* self"?
                #f"-D FLAG_DEBUG",
                #f"-D FLAG_PRINTD",
                #f"-D FLAG_ASSERTD",
            ] + PRECONFIGURED_CFLAGS,
        ),
        Extension(
            # the qualified name of the extension module to build
            'spkmeansmodule',
            # the files to compile into our module relative to ``setup.py``
            [
                f"{PATH_SRC}/generics/common_utils.c",
                f"{PATH_SRC}/generics/matrix.c",
                f"{PATH_SRC}/generics/matrix_reader.c",
                f"{PATH_SRC}/algorithms/wam.c",
                f"{PATH_SRC}/algorithms/ddg.c",
                f"{PATH_SRC}/algorithms/lnorm.c",
                f"{PATH_SRC}/algorithms/jacobi.c",
                f"{PATH_SRC}/algorithms/eigengap.c",
                f"{PATH_SRC}/spkmeans.c",
                f"{PATH_SRC}/kmeans.c",
                f"{PATH_SRC}/spkmeansmodule.c"
            ],
            extra_compile_args=[
                "-Wall", "-Wextra", "-Werror", "-lm",
                #"-pedantic-errors"
                "-Wno-error=missing-field-initializers", # FIXME - Issue in Python 3.8: https://github.com/SELinuxProject/setools/issues/31
                "-Wno-error=unused-parameter", # FIXME - what do I do with "PyObject* self"?
                #f"-D FLAG_DEBUG",
                #f"-D FLAG_PRINTD",
                #f"-D FLAG_ASSERTD",
                #f"-D FLAG_DEBUG_REVERSE_SORTING_INDICES"
            ] + PRECONFIGURED_CFLAGS,
        ),
    ],
    include_dirs=[
        f'{PATH_SRC}'
    ],
    cmdclass={"build":BuildCommand}
)
