import os
from results import res

def execute_python_file(file_path):
    try:
        with open(file_path, 'r') as file:
            python_code = file.read()
            exec(python_code)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")

file_path = 'train_model.py'
execute_python_file(file_path)
res()