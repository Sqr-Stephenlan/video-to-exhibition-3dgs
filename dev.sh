#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PY="$VENV_DIR/bin/python"

find_system_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No system Python found. Install Python 3 before bootstrapping." >&2
  return 1
}

doctor() {
  echo "Project root: $ROOT"
  echo "Venv dir: $VENV_DIR"
  if [ -x "$PY" ]; then
    echo "Python: $($PY -c 'import sys; print(sys.executable)')"
    echo "Version: $($PY --version 2>&1)"
    echo "Prefix: $($PY -c 'import sys; print(sys.prefix)')"
  elif [ -d "$VENV_DIR" ]; then
    echo "Venv status: present but $PY is not executable"
  else
    echo "Venv status: missing"
  fi

  for marker in uv.lock requirements.txt pyproject.toml poetry.lock Pipfile; do
    if [ -f "$marker" ]; then
      echo "Found: $marker"
    fi
  done
}

bootstrap() {
  if [ -x "$PY" ]; then
    echo "Existing virtual environment found at $VENV_DIR"
    doctor
    return 0
  fi

  if [ -d "$VENV_DIR" ]; then
    echo "$VENV_DIR exists but $PY is not executable; refusing to overwrite it." >&2
    exit 1
  fi

  if [ -f "uv.lock" ] && command -v uv >/dev/null 2>&1; then
    echo "Bootstrapping with uv sync"
    uv sync
    doctor
    return 0
  fi

  SYSTEM_PY="$(find_system_python)"
  "$SYSTEM_PY" -m venv "$VENV_DIR"
  "$PY" -m pip install --upgrade pip

  if [ -f "requirements.txt" ]; then
    "$PY" -m pip install -r requirements.txt
  elif [ -f "pyproject.toml" ]; then
    "$PY" -m pip install -e .
  else
    echo "No dependency file found; created $VENV_DIR only."
  fi

  doctor
}

ensure_venv() {
  if [ -x "$PY" ]; then
    return 0
  fi
  echo "No usable virtual environment found at $VENV_DIR." >&2
  echo "Run './dev.sh bootstrap' after confirming environment setup is desired." >&2
  exit 1
}

cmd="${1:-doctor}"
case "$cmd" in
  doctor)
    doctor
    ;;
  bootstrap)
    bootstrap
    ;;
  python)
    shift
    ensure_venv
    exec "$PY" "$@"
    ;;
  reconstruct)
    shift
    ensure_venv
    exec "$PY" -m scripts.longsplat.reconstruct_pipeline "$@"
    ;;
  pip)
    shift
    ensure_venv
    exec "$PY" -m pip "$@"
    ;;
  pytest)
    shift
    ensure_venv
    exec "$PY" -m pytest "$@"
    ;;
  mypy)
    shift
    ensure_venv
    exec "$PY" -m mypy "$@"
    ;;
  ruff)
    shift
    ensure_venv
    exec "$PY" -m ruff "$@"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: ./dev.sh {doctor|bootstrap|python|reconstruct|pip|pytest|mypy|ruff} ..." >&2
    exit 2
    ;;
esac
