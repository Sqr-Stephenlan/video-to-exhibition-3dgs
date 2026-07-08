<!-- codex-python-env:start -->
## Python environment

This project uses a local Python environment.

Always run Python commands through:

```bash
./dev.sh python ...
./dev.sh pip ...
./dev.sh pytest ...
```

Do not use bare `python`, `pip`, `pytest`, `mypy`, or `ruff`.

If `.venv` exists, use it. Do not create another virtual environment.
If `.venv` is missing, run `./dev.sh bootstrap` only after confirmation.
<!-- codex-python-env:end -->
