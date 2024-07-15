import os, json

def loadConfig(config_info):
    assert isinstance(config_info, str)
    # Check if it is a path to a JSON file
    if os.path.exists(config_info) and config_info.lower().endswith(".json"):
        # load json file
        with open(config_info, "r") as f:
            return json.load(f)
    else:
        return json.loads(config_info)
