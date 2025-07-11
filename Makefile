install_hooks:
	-uv add nbdev
	-uv add pre-commit
	-pre-commit install

install_jupyter:
	-uv add jupyter
	-uv add notebook
	-uv add ipython
	-uv add ipykernel
	-uv add ipywidgets

install_python_basics:
	-uv add umap-learn
	-uv add numpy
	-uv add pandas
	-uv add tables 
# ^ important dependecy for pandas!
	-uv add matplotlib
	-uv add watermark
	-uv add autoreload
	-uv add pathlib
	-uv add distro
	-uv add jupyter_black
	-uv add black
	-uv add memory_profiler
# add more packages!
