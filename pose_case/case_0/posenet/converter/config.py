import yaml
import os

BASE_DIR = os.path.dirname(__file__)

def load_config(config_name='config.yaml'):
    print(BASE_DIR)
    config_path = os.path.join(BASE_DIR, config_name)
    
    try:
        with open(config_path, "r") as cfg_f:
            cfg = yaml.safe_load(cfg_f)
        return cfg
    except FileNotFoundError:
        print(f"Configuration file '{config_name}' not found in '{BASE_DIR}'.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

# Example usage
if __name__ == "__main__":
    config = load_config()
    if config:
        print(config)
