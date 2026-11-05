"""Convierte un script con celdas '# %%' / '# %% [markdown]' en un notebook .ipynb y lo ejecuta.

Uso: python construir_notebook.py entrenamiento_clasificador.py
El notebook resultante queda guardado con todas sus salidas (tablas y gráficos), listo para verse en GitHub.
"""
import sys, re, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

src = Path(sys.argv[1]); text = src.read_text(encoding='utf-8')
cells = []
for block in re.split(r'^# %%', text, flags=re.M)[1:]:
    head, _, body = block.partition('\n')
    if head.strip().startswith('[markdown]'):
        md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip('\n').splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip('\n').replace("matplotlib.use('Agg')\n", '')))
nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
ExecutePreprocessor(timeout=1800, kernel_name='python3').preprocess(nb, {'metadata': {'path': str(src.parent)}})
out = src.with_suffix('.ipynb'); nbformat.write(nb, out); print('Notebook ejecutado y guardado en', out)
