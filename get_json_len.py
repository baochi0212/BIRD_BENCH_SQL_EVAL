import sys
import json

def get_file_length(file_path):
    if True:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return len(data)
    # except FileNotFoundError:
    #     print(f"File {file_path} not found.")

    #     return None
    # except json.JSONDecodeError:
    #     print(f"Invalid JSON format in file '{file_path}'.")
    #     return None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file>")
    else:
        json_file = sys.argv[1]
        length = get_file_length(json_file)
        if length is not None:
            print(length)
