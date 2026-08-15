#!/bin/zsh
set -e

APP_DIR="${0:A:h}"
cd "${APP_DIR}"

if [[ ! -x .venv/bin/python || ! -d node_modules ]]; then
  ./setup-local.sh
fi

./start-local.sh
