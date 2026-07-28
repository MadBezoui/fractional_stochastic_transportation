import os
import hashlib

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

files_to_hash = []
for root, dirs, files in os.walk('computational_validation'):
    if 'venv' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.csv') or file.endswith('.log'):
            files_to_hash.append(os.path.join(root, file))

for root, dirs, files in os.walk('manuscript'):
    for file in files:
        if file in ['main2.tex', 'main2.pdf', 'tables_output.txt']:
            files_to_hash.append(os.path.join(root, file))

files_to_hash.append('generate_all_tables.py')

files_to_hash = list(set(files_to_hash))
files_to_hash.sort()

with open('SHA256SUMS.txt', 'w') as f:
    for filepath in files_to_hash:
        f.write(f"{hash_file(filepath)}  {filepath}\n")

with open('MANIFEST.txt', 'w') as f:
    f.write("Fractional Stochastic Transportation - Supplementary Archive\n")
    f.write("============================================================\n\n")
    f.write("This archive contains the source code, logs, and generated tables for the computational study.\n\n")
    f.write("Files included:\n")
    for filepath in files_to_hash:
        f.write(f"- {filepath}\n")
    f.write("\nTo reproduce the tables from the manuscript, run:\n")
    f.write("    python generate_all_tables.py\n")

print("Manifests generated.")
