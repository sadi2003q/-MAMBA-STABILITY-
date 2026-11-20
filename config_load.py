import yaml


def load_config(path="config.yaml"):
    # Step 1: open the settings file for reading
    with open(path, "r") as file:
        # Step 2: turn the YAML text into a Python dictionary
        config = yaml.safe_load(file)

    return config