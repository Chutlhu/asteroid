from pathlib import Path
import json

N = 200

basedir = "data/wav8k/min/"
basedir_path = Path(basedir)
newdir = f"data/wav8k/min{N}/"


print(basedir_path.exists())
if not basedir_path.exists():
    raise ValueError("Wrong folder man!")

for file in basedir_path.rglob('*.*'):
    print(file)

    # get the first N elements
    with open(file) as f:
        wavlist = json.load(f)
    wavlist = wavlist[:N]
    
    # recover the base name
    fullfile = str(file.resolve())
    fullfile = Path(fullfile.replace(basedir,newdir))
    
    # create the directoty
    dir = fullfile.parents[0]
    dir.mkdir(exist_ok=True,parents=True)
    print(dir)
    # write file
    with open(fullfile, 'w') as f:
        json.dump(wavlist, f, indent=4)