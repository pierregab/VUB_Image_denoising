# a script to clear the runs folder (and debug inside it)

import os
import shutil

# Define the path to the runs folder
runs_folder = 'runs/paper_gan'

# Clear the runs folder
shutil.rmtree(runs_folder, ignore_errors=True)
os.makedirs(runs_folder, exist_ok=True)
print(f"Deleted and recreated the runs folder at {runs_folder}")