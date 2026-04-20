import hashlib

import numpy as np
from pathlib import Path


def dict_to_id(result_info: dict) -> str:
    """Generates a unique ID from the result_info dictionary."""
    values = str(sorted(result_info.items()))
    result_id = hashlib.sha256(values.encode()).hexdigest()
    return result_id

def init_results_folder(path: str = "./data/results"):
    """Initializes the results folder if it does not exist."""
    # check if the folder exists, if not, create it
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Results folder created at {path}")
    else:
        print(f"Results folder already exists at {path}")

def check_exisiting_results(path: str = "./data/results", result_id: str = None) -> bool:
    """Checks if results already exist for the given result_id."""
    results_path = Path(path) / result_id
    return results_path.exists()

def save_results(path: Path = "./data/results",
                 model_info: dict = None,
                 result_info: dict = None,
                 result_data: dict = None,
                 verbose: bool = True,
                 force_save: bool = False):
    """Saves the training results to a file."""

    # i want to add a prompt to add potential comments on a set of result that are not clear of writen in the standars informations

    # Generate a UUid from the result_data to ensure that we do not save the same results twice


    result_id = dict_to_id(model_info)

    results_path = path + "/" + result_id

    results_path = Path(results_path)

    # check if the folder exists, if it exists, make a new one with a number at the end
    results_exists = False
    if results_path.exists() and force_save:
        # create a new folder with a number at the end
        i = 1
        temp_path = results_path.parent / f"{results_path.name}_{i}/"
        while temp_path.exists():
            i += 1
            temp_path = results_path.parent / f"{results_path.name}_{i}/"
        results_path = temp_path
        if verbose:
            print(f"Results already exist with the same parameters. Saving to {results_path} instead.")
    elif results_path.exists() and not force_save:
        if verbose:
            print("Results already exist with the same parameters. Not saving again.")
        return
    
    results_path.mkdir()


    """
    Save the result to a folder with the following format:
    results/{data}/{embedding}/
    with files:
    - details.txt
    - data.csv
    """
    # Save details.txt
    with open(results_path / "details.txt", "w") as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    
    with open(results_path / "results_info.txt", "w") as f:
        for key, value in result_info.items():
            f.write(f"{key}: {value}\n")

    for key, value in result_data.items():
        # Save data.csv
        np.savetxt(results_path / f"{key}.csv", value, delimiter=",")
        if verbose:
            print(f"{key}.csv saved to {results_path}")

    if verbose:
        print(f"------Results saved to {results_path}------")


def read_results(path : Path = None, 
                 model_info: dict = None,
                 number: int = None,
                 verbose: bool = True):
    
    """Reads the training results from a folder."""
    result_id = dict_to_id(model_info)

    results_path = path + "/" + result_id
    results_path = Path(results_path)
    if not results_path.exists():
        if verbose:
            print(f"No results found for the given model_info at {results_path}.")
        return None
    
    result_info = {}
    with open(results_path / "results_info.txt", "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            result_info[key] = value

    data = {}
    for file in results_path.glob("*.csv"):
        key = file.stem
        data[key] = np.loadtxt(file, delimiter=",")

    return result_info, data

