import json

def generate_json_config(all_optimal_lags: dict[str, dict[str, int]], file_name: str) -> None:
    """
        Given the output from the aggregate_lags() function, create either a json object for easy viewing & presenting
    """
    file_path = f'{file_name}.json'

    with open(file_path, 'w') as json_file:
        json.dump(all_optimal_lags, json_file, indent=4) # indenting by 4; its the python standard

    print(f"Data successfully saved to {file_path}")
