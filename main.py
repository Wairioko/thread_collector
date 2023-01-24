import sys

filenames = ['grab_data.py', 'clean_data.py', 'model.py', 'create_url.py']

for filename in filenames:
    exec(open(filename).read())
sys.exit()
