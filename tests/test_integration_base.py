import unittest
from abc import abstractmethod, abstractclassmethod
from typing import List, Callable, Tuple, Union, Optional, Any
NoneType = type(None)
from math import inf
import time
import subprocess
import os
import sys
import numpy as np
from sklearn.datasets import make_blobs
import random
import inspect
from subprocess import CalledProcessError

default_path_to_writable_folder = "/tmp"

class TestIntegrationBase():

    path_to_repo_folder: str = None
    path_to_writable_folder: str = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_workdir()
        os.environ['PATH_SRC'] = f"{cls.path_to_workdir}"
        os.environ['PATH_OUT'] = f"{cls.path_to_workdir}"
        cls.path_to_executable: str = dict()
        cls.compile()

    @classmethod
    def tearDownClass(cls) -> None:
        #os.system(f"rm -rf {cls.path_to_workdir}")
        pass

    @classmethod
    def setup_workdir(cls) -> None:
        if type(cls.path_to_writable_folder) not in {str, NoneType}:
            raise ValueError("Invalid type - path_to_writable_folder should be a string or None")
        cls.path_to_writable_folder = cls.path_to_writable_folder or default_path_to_writable_folder
        cls.path_to_workdir = f"{cls.path_to_writable_folder}/test_spkmeans_{int(time.time())}"
        os.makedirs(f"{cls.path_to_workdir}", exist_ok=False)
        subprocess.check_output(
            ["bash", f"{cls.path_to_repo_folder}/submit.sh"],
            cwd=cls.path_to_repo_folder,
            env=dict(os.environ, SAVEDIR=cls.path_to_workdir, REPODIR=cls.path_to_repo_folder)
        )
        os.chdir(cls.path_to_workdir)
        os.system(f"unzip *.zip; rm *.zip")
        os.system(f"mv */* .")
        sys.path.insert(0, cls.path_to_workdir)
        os.system(f"cp {cls.path_to_workdir}/*.so .")

    @abstractclassmethod 
    def compile(cls) -> None:
        raise NotImplementedError("Override using compile_c_standalone, compile_c_python or both")
    
    @classmethod
    def compile_c_standalone(cls) -> str:
        # compile c (standalone)
        # a nonzero exit code will yield CalledProcessError
        subprocess.check_output(["./comp.sh"], cwd=cls.path_to_workdir)
        return f"{cls.path_to_workdir}/spkmeans"

    @classmethod
    def compile_c_python(cls) -> str:
        # compile c (python module)
        subprocess.check_output(
            ["python3", "setup.py", "build_ext", "--inplace"],
            cwd=cls.path_to_workdir,
            env=dict(os.environ, FLAG_DEBUG='1'))
        import spkmeansmodule
        cls.spkmeansmodule = spkmeansmodule
        import mykmeanssp
        cls.kmeans_pp = mykmeanssp
        return f"{cls.path_to_workdir}/spkmeans.py"
    
    @classmethod
    def run_with_data(cls,
            path_to_executable: str,
            k: Optional[int], goal: str, data: List[List[float]]) -> str:
        return run_with_data(path_to_executable, k, goal, data, path_to_data=f"{cls.path_to_workdir}/{goal}.txt")
    
    @classmethod
    def run_with_args(cls,
            path_to_executable: str,
            *extras: Optional[List[str]]) -> str:
        return run_with_args(path_to_executable, *extras, path_to_workdir=cls.path_to_workdir)

    def assert_mat_dist(self, real: np.ndarray, calc: np.ndarray, comments="undefined"):
        real, calc = np.round(np.array(real),4), np.round(np.array(calc),4)
        self.assertEqual(real.shape, calc.shape)
        dist = min(
            np.mean(np.abs(real-calc)) / np.mean(np.abs(real)),
            np.mean(np.square(real-calc))
        )
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        caller = calframe[1][3]
        self.assertLess(dist, 1e-5, f"workdir is {self.path_to_workdir}, comments: \"{comments}\"\ncaller is {caller}\nREAL:\n{real}\n\nFAKE:\n{calc}")


def run_with_data(
        path_to_executable: str,
        k: Optional[int], goal: str, data: List[List[float]],
        path_to_data: str) -> str:
    if (type(path_to_executable) is not str) \
            or (type(k) not in {int, str, NoneType}) \
            or (type(goal) not in {str}) \
            or (type(data) not in {list}) \
            or (type(path_to_data) not in {str}):
        print(f"{path_to_executable},,, {k},,, {goal},,, {data},,, {path_to_data}")
        raise ValueError("run_with_data() found args with invalid types - aborting")
    file_write_mat(path_to_data, data)
    args = []
    if k != None: args.append(str(k))
    args.append(goal)
    args.append(path_to_data)
    result_string = run_with_args(path_to_executable, *args)
    return result_string


def run_with_valid_args(
        path_to_executable: str,
        k: Optional[int], goal: str, path_to_data: str) -> str:
    if (type(path_to_executable) is not str) \
            or (type(k) not in {int, str, NoneType}) \
            or (type(goal) not in {str}) \
            or (type(path_to_data) not in {str}):
        raise ValueError("run_with_valid_args() found args with invalid types - aborting")
    args = []
    if k != None: args.append(str(k))
    args.append(goal)
    args.append(path_to_data)
    result_string = run_with_args(path_to_executable, k, goal, path_to_data)
    return result_string

def run_with_args(
        path_to_executable: str, *extras: Optional[List[str]], **kwargs) -> str:
    path_to_workdir = kwargs.get('path_to_workdir', None)
    if (type(path_to_executable) is not str) \
            or (type(extras) not in {tuple, NoneType}) \
            or (type(path_to_workdir) not in {str, NoneType}):
        print(f"{path_to_executable},,, {extras},,, {path_to_workdir}")
        raise ValueError("run_with_args() found args with invalid types - aborting")
    if path_to_workdir: os.chdir(path_to_workdir)
    args = []
    if path_to_executable.endswith(".py"):
        args.append("python3")
    args.append(path_to_executable)
    args += extras
    timeout = kwargs.get('timeout', inf)
    # FIXME - implement timeout maybe
    try:
        result = subprocess.check_output(args)
        result_string = result.decode('utf-8')
        return result_string
    except CalledProcessError as e:
        raise Exception(f"Got CalledProcessError with {e.output}\n\tRun args were:\n\t{' '.join(e.cmd)}\n\t")
    
def file_write_mat(path: str, mat: List[List[float]]) -> None:
    with open(path, 'w') as f:
        mat_str = mat_to_str(mat)
        f.write(mat_str)

def file_read_mat(path: str) -> List[List[float]]:
    with open(path, 'r') as f:
        mat_str = f.read()
        return str_to_mat(mat_str)

def mat_to_str(mat: List[List[float]]) -> str:
    return '\n'.join([','.join([str(y) for y in x]) for x in mat]) + '\n'

def str_to_mat(mat_str: str) -> List[List[float]]:
    #print(f"caught mat str:\n{mat_str}\nmat_str_end")
    def filter_out_junk_chars(s:str):
        t = ""
        for c in s:
            if c in {',','.','\n','-','+','0','1','2','3','4','5','6','7','8','9'}:
                t += c
        return t
    #mat_str = filter_out_junk_chars(mat_str)
    mat = mat_str.split("\n")
    mat = [line.strip() for line in mat if line.strip() != ""]
    mat = [line.split(",") for line in mat]
    mat = [[float(y) for y in x] for x in mat]
    return mat

def make_compatible_blob(n=100,d=10, offset=0) -> List[List[float]]:
    x, _ = make_blobs(n_samples=n, n_features=d)
    x: np.ndarray
    return [[float(z+offset) for z in y] for y in list(x)]

def make_compatible_blob_symmetric(n=100) -> List[List[float]]:
    a = np.random.rand(n,n)
    bottom_left = np.tril(a)
    top_right = np.tril(a,-1).T
    m = bottom_left + top_right
    return [[float(x) for x in y] for y in m]

def random_blob(size: str = None):
    if not size:
        size = random.choice(['max','big','small','tiny'])
    if size=='max':
        return make_compatible_blob(1000,10)
    elif size=='big':
        return make_compatible_blob(int(np.random.randint(969,1010)),int(np.random.randint(8,69)))
    elif size == 'small':
        return make_compatible_blob(int(np.random.randint(8,13)),int(np.random.randint(3,6)))
    elif size == 'tiny':
        return make_compatible_blob(5,2)
    else:
        raise ValueError()

def random_blob_symmetric(size: str = None):
    if not size:
        size = random.choice(['big','small','tiny'])
    if size=='big':
        return make_compatible_blob_symmetric(int(np.random.randint(969,1010)))
    elif size == 'small':
        return make_compatible_blob_symmetric(int(np.random.randint(8,13)))
    elif size == 'tiny':
        return make_compatible_blob_symmetric(int(np.random.randint(2,5)))
    else:
        raise ValueError()
