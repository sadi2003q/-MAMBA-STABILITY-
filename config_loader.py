import os

import yaml


def load_config(file_name="config.yaml"):
    # Step 1: find the folder this file (config_loader.py) lives in
    this_file = os.path.abspath(__file__)
    project_folder = os.path.dirname(this_file)

    # Step 2: build the full path to config.yaml inside that folder
    # (so it works even if you run a script from another folder)
    config_path = os.path.join(project_folder, file_name)

    # Step 3: stop with a clear message if the file is missing
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find the settings file: {config_path}")

    # Step 4: open the file and turn the YAML text into a Python dictionary
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    # Quick check: run  python config_loader.py  and look at what was read
    config = load_config()

    print("Sections found in config.yaml:")
    for section_name in config:
        print("  ", section_name, "->", config[section_name])