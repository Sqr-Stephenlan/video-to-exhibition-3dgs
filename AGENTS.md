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

## Project Codex prompts

Before planning or implementing the video preprocessing feature, read:

```text
.codex/prompts/preprocess_video_feature.md
.codex/status/preprocess_video_feature.md
```

Treat the prompt file as the project instructions for this feature, and treat
the status file as the source of truth for current phase, completed work, next
steps, and open decisions. Update the status file after meaningful project
progress. Keep generated videos, frames, manifests, model outputs, dependency
source trees, caches, and temporary Codex scratch files out of git unless the
prompt explicitly says otherwise.
