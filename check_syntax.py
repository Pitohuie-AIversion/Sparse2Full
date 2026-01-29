import ast
import sys

filename = 'tools/training/train_real_data_ar.py'

try:
    with open(filename, 'r') as f:
        source = f.read()
    ast.parse(source)
    print("Syntax is valid.")
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}")
    print(e.text)
except Exception as e:
    print(f"Error: {e}")
