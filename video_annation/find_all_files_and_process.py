import os
import json
import subprocess

def find_json_files(directory):
    """
    Recursively find all JSON files in the given directory.
    """
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    directory = config['directory']
    script_path = config['script_path']
    
    json_files = find_json_files(directory)
    for json_file in json_files:
        command = f'{script_path} {json_file}'
        print(f'Executing: {command}')
        subprocess.run(['python', script_path, json_file], check=True)

if __name__ == "__main__":
    import os
    print("当前工作目录:", os.getcwd())
    main()
