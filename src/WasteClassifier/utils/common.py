# Standard library imports
import os  
import json  
import base64  
from pathlib import Path  
import pickle  
from typing import Any  
from textwrap import dedent

# Third-party library imports
import yaml  
from box.exceptions import BoxValueError  
from box import ConfigBox  
from ensure import ensure_annotations  

# Local application/library imports
from WasteClassifier import logger  


@ensure_annotations
def read_yaml(yamlPath: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        yamlPath (Path): The path to the YAML file.

    Returns:
        ConfigBox: A ConfigBox object containing the parsed YAML content.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: If there is an error reading the YAML file.
    """

    try:
        with open(yamlPath, "r") as yamlFile:
            content = yaml.safe_load(yamlFile)
            logger.info(f"yaml file: {yamlPath} loaded successfully!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.error(f"Error reading yaml file: {yamlPath}\n Error: {e}")
        raise


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(jsonPath: Path) -> ConfigBox:
    """
    Reads a JSON file and returns its content as a ConfigBox object.

    Args:
        jsonPath (Path): The path to the JSON file.

    Returns:
        ConfigBox: A ConfigBox object containing the parsed JSON content.

    Raises:
        ValueError: If the JSON file is empty.
        Exception: If there is an error reading the JSON file.
    """
    try:
        with open(jsonPath) as f:
            content = json.load(f)

        logger.info(f"json file loaded from: {jsonPath}")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError("json file is empty")
    except Exception as e:
        logger.error(f"Error reading json file: {jsonPath}\n Error: {e}")
        raise


@ensure_annotations
def save_bin_file(data: Any, binPath: Path):
    """
    Saves the given data to a binary file at the given path.

    Args:
        data (Any): The data to be saved to the binary file.
        binPath (Path): The path where the binary file should be saved.
    """
    pickle.dump(data, open(binPath, 'wb'))
    logger.info(f"bin file saved at: {binPath}")

@ensure_annotations
def load_bin_file(binPath: Path) -> Any:
    """
    Loads a binary file from the given path and returns its content.

    Args:
        binPath (Path): The path to the binary file to load.

    Returns:
        Any: The content of the binary file.

    Raises:
        FileNotFoundError: If the binary file does not exist.
        Exception: If there is an error loading the binary file.
    """
    try:
        with open(binPath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"bin file loaded from: {binPath}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"bin file not found at: {binPath}")
    except Exception as e:
        logger.error(f"Error loading bin file: {binPath}\n Error: {e}")
        raise


@ensure_annotations
def create_directories(dirPaths: list, verbose=True):
    """
    Creates one or more directories given a list of directory paths.

    Args:
        dirPaths (list): A list of directory paths to create.
        verbose (bool, optional): If True, will log a message after creating each directory. Defaults to True.
    """
    for path in dirPaths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in KB.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"file size: ~ {size_in_kb}"


def decode_image(imgString: str, fileName: str) -> None:
    """
    Decodes a base64 encoded image string and saves it to a file.

    Args:
        imgstring (str): The base64 encoded image string.
        fileName (str): The file name to save the image as.

    Returns:
        None
    """
    imgdata = base64.b64decode(imgString)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encode_image_into_base64(imagePath: Path) -> str:
    """
    Encodes an image file into a base64 string.

    Args:
        imagePath (Path): The path to the image file to encode.

    Returns:
        base64: The base64 encoded string of the image file.
    """
    with open(imagePath, "rb") as f:
        return base64.b64encode(f.read())


@ensure_annotations
def start_stage_logger(stage_name: str, length: int = 40, symbol: str = "#") -> str:

    # ANSI escape codes for color
    GREEN = '\033[92m'  # Green
    RESET = '\033[0m' 
    stage_name_start = "".join([stage_name, " started"])
    
    stage_start = dedent(f"""\
        
        {GREEN}{length * symbol}
        {stage_name_start.upper().center(length," ")}
        {length * symbol}{RESET}
            
        """)
    return stage_start


@ensure_annotations
def end_stage_logger(stage_name: str, length: int = 40, symbol: str = "#") -> str:

    # ANSI escape codes for color
    GREEN = '\033[32m'  # Green
    RESET = '\033[0m' 
    
    stage_name_start = "".join([stage_name, " ended"])
    
    stage_end = dedent(f"""\

        
        {GREEN}{length * symbol}
        {stage_name_start.upper().center(length," ")}
        {length * symbol}{RESET}
            
        """)
    return stage_end

def save_to_pickle(obj: object, path: Path, file_name: str) -> None:
    """
    Safely saves an object to a pickle file.

    Args:
        obj (object): The object to save.
        path (Path): The path to the directory where the pickle file should be saved.
        file_name (str): The name of the pickle file, without the extension.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(path / file_name, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(path: Path, file_name: str) -> object:
    """
    Loads an object from a pickle file.

    Args:
        path (Path): The path to the directory where the pickle file is located.
        file_name (str): The name of the pickle file, without the extension.

    Returns:
        object: The loaded object.
    """
    with open(path / file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj
