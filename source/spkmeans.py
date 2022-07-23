import os
import sys
from typing import List
import numpy as np
import pandas as pd
import mykmeanssp
from kmeans_pp import calc_kmeanspp
from utils import *
from definitions import *


def main():
    np.random.seed(0)
    k, goal, datapoints = get_data_from_cmd()
    if   goal == 'spk':
        results = calc_kmeanspp(k, datapoints)
    elif goal == 'wam':
        results = calc_wam(datapoints)
    elif goal == 'ddg':
        results = calc_ddg(datapoints)
    elif goal == 'lnorm':
        results = calc_L_norm(datapoints)
    elif goal == 'jacobi':
        eigenvalues, eigenvectors = jacobi_algorithm(datapoints)
        print(','.join(["%.4f"%y for y in x]) for x in eigenvalues)
        results = eigenvectors
        pass
    print('\n'.join([','.join(["%.4f"%y for y in x]) for x in results]))


def get_data_from_cmd():

    def _get_raw_cmd_args():
        args = sys.argv
        if not args:
            raise InvalidInputTrigger("Args are empty!")
        if args[0] in ["python", "python3", "python.exe", "python3.exe"]:
            args = args[1:]
            if not args:
                raise InvalidInputTrigger("Args are empty after removing \"python\" prefix")
        if args[0][-3:] == ".py":
            args = args[1:]
            if not args:
                raise InvalidInputTrigger("Args are empty after removing executable filename prefix")
        return args

    def _get_cmd_args():
        args = _get_raw_cmd_args()
        try:
            if len(args) == 2: # N/A, goal, file_name
                return None, args[0], args[1]
            if len(args) == 3: # k, goal, file_name
                return int(args[0]), args[1], args[2]
            else:
                raise InvalidInputTrigger("Cannot parse input format - number of args must be in {3}")
        except:
            raise InvalidInputTrigger("An error has occurred while parsing CLI input - one of the arguments is NaN")

    def _validate_input_filename(file: str):
        if not os.path.exists(file):
            raise InvalidInputTrigger(f"Specified path does not exist - \"{file}\"")
        if not (file.lower().endswith("csv") or file.lower().endswith("txt")):
            raise InvalidInputTrigger(f"Specified path does not end in a permitted extension - \"{file}\"")
    
    def _read_data_as_np(file_name: str) -> np.ndarray:
        _validate_input_filename(file_name)
        path_file = os.path.join(os.getcwd(), file_name)
        df = pd.read_csv(path_file, header=None).rename({0: "index"}, axis=1)
        df_sorted = df.sort_values('index')
        #df_sorted.drop('index', inplace=True, axis=1)
        data = df_sorted.to_numpy()
        return data

    def _verify_params_make_sense(k: int, goal: str, data: np.ndarray):
        assert(data.ndim == 2)
        n, d = data.shape
        if n == 0:
            raise GenericErrorTrigger("Data, as parsed, is empty - nothing to work on")
        if goal not in {'spk', 'wam', 'ddg', 'lnorm', 'jacobi'}:
            raise InvalidInputTrigger(f"Unrecognized goal specified - {goal}")
        if goal in {'spk'}:
            if k == None:
                raise InvalidInputTrigger(f"Only 2 parameters specified, expected integer k for goal {goal}")
            if not (0 < k < n):
                raise InvalidInputTrigger("The following must hold: 0 < k < n, but k={k} and n={n}")
            if (d - 1) < 1:
                raise GenericErrorTrigger("Datapoints number of dimensions must be at least 1, not including dimension 0 (index)")
            if any([(not first_indice.is_integer()) or (not (0 <= int(first_indice) < n)) \
                    for first_indice in data[:,0]]):
                raise GenericErrorTrigger(f"One of the datapoints is missing a valid index in its first dimension")
        elif goal in {'wam', 'ddg', 'lnorm'}:
            if k != None:
                raise InvalidInputTrigger(f"3 parameters specified, did not expect k for goal {goal}")
            if d < 1:
                raise GenericErrorTrigger("Datapoints number of dimensions must be at least 1")
        elif goal in {'jacobi'}:
            if k != None:
                raise InvalidInputTrigger(f"3 parameters specified, did not expect k for goal {goal}")
            if d != n:
                raise GenericErrorTrigger(f"Jacobi expects a symmetric matrix, but n != d ({n} != {d})")
    
    k, goal, file_name = _get_cmd_args()
    datapoints = _read_data_as_np(file_name)
    _verify_params_make_sense(k, goal, datapoints)
    return k, goal, datapoints


def exit_gracefully_with_err(err: Exception):
    def exit_gracefully_with_err_string(msg: str):
        print(msg)
        exit(1)
    error_to_string_map = {} if FLAG_VERBOSE_ERRORS else {
        InvalidInputTrigger: MSG_ERR_INVALID_INPUT       ,
        GenericErrorTrigger: MSG_ERR_GENERIC             }
    exit_gracefully_with_err_string(error_to_string_map.get(type(err),
        f"Exiting gracefully with an unexplained error:\n{str(err)}" if FLAG_VERBOSE_ERRORS \
        else MSG_ERR_GENERIC))


if __name__ == '__main__':
    if not CHANGE_TO_TRUE_BEFORE_ASSIGNMENT:
        print("===== WARNING =====")
        print("      MAKE SURE YOU CHANGE CHANGE_TO_TRUE_BEFORE_ASSIGNMENT TO TRUE BEFORE ASSIGNMENT")
        print("      OTHERWISE, YOU WILL GET DEBUG BEHAVIOR")
    try:
        main()
    except Exception as e:
        exit_gracefully_with_err(e)
