from setuptools import setup, find_packages, Extension
import sysconfig

PATH_SRC=".././source/c"
PATH_OUT=".././output"

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += [
    "-Wall", "-Werror", "-pedantic-errors",
    
]

# setup() parameters - https://packaging.python.org/guides/distributing-packages-using-setuptools/
setup(
    name='KMeansAlgorithm',
    version='6.9.4.2.0',
    author="Spongebob Squarepants",
    author_email="a@xb.ax",
    description="This is an implementation of the KMeans algorithm",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        # We need to tell the world this is a CPython extension
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
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
                f"{PATH_SRC}/kmeans.c"
            ],
            extra_compile_args=extra_compile_args+[
                f"-D FLAG_DEBUG",
                f"-D FLAG_PRINTD",
                f"-D FLAG_ASSERTD",
                f"-I {PATH_SRC}"
            ],
        ),
    ]
)
