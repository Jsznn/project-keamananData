import yaml
from pathlib import Path

def load_config(config_path="config/config.yml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        # Try looking relative to the project root if running from src or notebooks
        # This is a simple heuristic; for production code, using environment variables or a more robust path resolution is better.
        project_root = Path(__file__).parent.parent
        path = project_root / config_path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path} or {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
