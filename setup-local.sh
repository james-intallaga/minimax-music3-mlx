#!/bin/zsh
set -eu

APP_DIR="${0:A:h}"
cd "${APP_DIR}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This local MLX edition requires an Apple Silicon Mac."
  exit 1
fi

node_supported() {
  "$1" -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 20 || (major === 20 && minor >= 9) ? 0 : 1)' 2>/dev/null
}

find_node() {
  local candidate
  for candidate in "${MINIMAX_MUSIC3_NODE_BIN:-}" "$(command -v node 2>/dev/null || true)" /opt/homebrew/bin/node /usr/local/bin/node; do
    if [[ -n "${candidate}" && -x "${candidate}" ]] && node_supported "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

NODE_BIN="$(find_node || true)"
if [[ -z "${NODE_BIN}" ]]; then
  echo "Node.js 20.9 or newer is required. Install the current LTS version from https://nodejs.org and run setup again."
  exit 1
fi
export PATH="${NODE_BIN:h}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if command -v uv >/dev/null 2>&1; then
  [[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
  uv pip sync --python .venv/bin/python backend/requirements.txt
else
  [[ -x .venv/bin/python ]] || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install --requirement backend/requirements.txt
fi

npm ci
npm run build
echo "Setup complete. Double-click Open MiniMax Music 3.command or run ./start-local.sh"
