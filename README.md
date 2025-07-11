# Research project
This is a template for the repository of a reseach project containing research code. It provides the folder structure and the pre-commit hooks, and it assumes you are using uv as your packaging manager.

1. After creating a new repo off this one, initialize a uv environment by running `uv init --python 3.12`. This creates the environment and adds all uv-related files to the repo.
2. Run `make install_hooks` to set up the pre-commit hooks.
3. Run `make install_jupyter` to get jupyter working.
4. Run `make install_python_basics` to install some python basic files.
5. For the installable package, a name has to be choosen, the `src/package_code` folder renamed, and the line `name = "package_code"` in `mypyproject.toml` file and notebook imports edited accordingly. Afterwards, in the uv environment, run `uv pip install -e .`.

To activate your uv environment, use `source .venv/bin/activate`.