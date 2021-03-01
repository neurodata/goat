jupytext --to notebook --output gmot/docs/$1.ipynb gmot/scripts/$1.py
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=-1 gmot/docs/$1.ipynb 
python gmot/docs/add_cell_tags.py gmot/docs/$1.ipynb
# {'metadata': {'path': run_path}}
# https://github.com/jupyter/nbconvert/blob/7ee82983a580464b0f07c68e35efbd5a0175ff4e/nbconvert/preprocessors/execute.py#L63
# --ExecutePreprocessor.record_timing=True