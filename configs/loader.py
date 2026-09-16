from pathlib import Path
import yaml

def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        path (str): The path to the YAML file.

    Returns:
        The parsed YAML content as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain yaml mapping: {path}")
    return config