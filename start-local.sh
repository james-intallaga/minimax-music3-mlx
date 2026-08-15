#!/bin/zsh
set -eu

APP_DIR="${0:A:h}"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"
WEB_HOST="127.0.0.1"
WEB_PORT="3000"
API_HOST="127.0.0.1"
API_PORT="7860"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${APP_DIR}/node_modules/.bin/next" ]]; then
  echo "Please run ./setup-local.sh first."
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
  echo "Node.js 20.9 or newer is required. Install the current LTS version from https://nodejs.org and try again."
  exit 1
fi
export PATH="${NODE_BIN:h}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

port_owner() {
  /usr/sbin/lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | tail -n +2 || true
}

for port in "${WEB_PORT}" "${API_PORT}"; do
  owner="$(port_owner "${port}")"
  if [[ -n "${owner}" ]]; then
    echo "MiniMax Music 3 cannot start because local port ${port} is already in use:"
    echo "${owner}"
    echo "Close that app and open MiniMax Music 3 again."
    exit 1
  fi
done

collect_tree() {
  local parent="$1"
  local child
  echo "${parent}"
  for child in $(/usr/bin/pgrep -P "${parent}" 2>/dev/null || true); do
    collect_tree "${child}"
  done
}

cleanup() {
  trap - EXIT INT TERM
  local targets=()
  local pid
  [[ -n "${WEB_PID:-}" ]] && targets+=("${(@f)$(collect_tree "${WEB_PID}")}")
  [[ -n "${API_PID:-}" ]] && targets+=("${(@f)$(collect_tree "${API_PID}")}")
  for pid in "${targets[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for attempt in {1..30}; do
    local running=false
    for pid in "${targets[@]}"; do
      kill -0 "${pid}" 2>/dev/null && running=true
    done
    [[ "${running}" == false ]] && break
    sleep 0.1
  done
  for pid in "${targets[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  [[ -n "${WEB_PID:-}" ]] && wait "${WEB_PID}" 2>/dev/null || true
  [[ -n "${API_PID:-}" ]] && wait "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export MINIMAX_MUSIC3_LOCAL_TOKEN="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_hex(32))')"
export MINIMAX_MUSIC3_WEB_ORIGIN="http://${WEB_HOST}:${WEB_PORT}"
export MINIMAX_MUSIC3_ENGINE_ORIGIN="http://${API_HOST}:${API_PORT}"

cd "${APP_DIR}"
"${PYTHON_BIN}" -m uvicorn backend.main:app --host "${API_HOST}" --port "${API_PORT}" --no-server-header &
API_PID=$!

for attempt in {1..40}; do
  if /usr/bin/curl --silent --fail --header "Authorization: Bearer ${MINIMAX_MUSIC3_LOCAL_TOKEN}" "${MINIMAX_MUSIC3_ENGINE_ORIGIN}/api/status" >/dev/null; then
    break
  fi
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "The local music engine stopped during startup."
    exit 1
  fi
  sleep 0.25
done

"${NODE_BIN}" "${APP_DIR}/node_modules/next/dist/bin/next" start --hostname "${WEB_HOST}" --port "${WEB_PORT}" &
WEB_PID=$!

for attempt in {1..40}; do
  if /usr/bin/curl --silent --fail "${MINIMAX_MUSIC3_WEB_ORIGIN}" >/dev/null; then
    open "${MINIMAX_MUSIC3_WEB_ORIGIN}"
    break
  fi
  if ! kill -0 "${WEB_PID}" 2>/dev/null; then
    echo "The local web app stopped during startup."
    exit 1
  fi
  sleep 0.25
done

echo "MiniMax Music 3 is running privately at ${MINIMAX_MUSIC3_WEB_ORIGIN}"
echo "Close this window or press Control-C to stop it."

while kill -0 "${API_PID}" 2>/dev/null && kill -0 "${WEB_PID}" 2>/dev/null; do
  sleep 1
done

echo "One part of MiniMax Music 3 stopped. Closing the local app safely."
exit 1
