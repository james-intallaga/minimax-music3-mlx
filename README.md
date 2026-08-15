# amma.live local music

Make complete songs privately on an Apple Silicon Mac with MiniMax‑Music3 and MLX. Prompts, lyrics, generated audio, and lyric synchronization stay on the computer; no amma.live account, API key, subscription, analytics, or cloud storage is used.

This repository contains an **MIT-licensed open-source app and MLX inference implementation**. MiniMax‑Music3 is an **open-weight model under the separate MiniMax‑Music3 Community License**; the model itself is not covered by this repository's MIT license.

New users can start with the plain-language [How to use amma.live guide](HOW_TO_USE.md).

## About amma.live

[amma.live](https://amma.live) supports the wider creator journey: making music, developing an artist identity, publishing responsibly, and sharing with listeners. This local edition is independent and does not connect to amma.live services.

## Hardware

- Apple Silicon Mac (M1, M2, M3, M4, or newer)
- macOS 14 or newer
- **64 GB unified memory is the supported minimum**
- **48 GB is experimental and may swap heavily or fail on longer songs**
- 32–36 GB is not supported by the current full-precision pipeline
- About 42 GB free storage for dependencies, approximately 29 GB of models, and songs
- Node.js 20.9 or newer and Python 3.12

The checkpoint is approximately 11.7 billion parameters, not 30 billion. Its persistent weights occupy roughly 28 GB before generation caches, working memory, macOS, the browser, and local lyric synchronization. A future quantized, staged-loading edition may target 24 GB Macs, but this release does not claim that support.

Five-minute songs are supported. On the development M3 Max, a five-minute song can take roughly 45–50 minutes; other Macs will vary considerably.

## One-click start

1. Download or clone this repository.
2. Install the current Node.js LTS release from [nodejs.org](https://nodejs.org).
3. Double-click **Open amma.live Music.command**.
4. On the first launch, click **Download MiniMax‑Music3**.

The setup installs pinned, integrity-checked dependencies and builds the local app. Later launches reuse the installation and model files. The launcher refuses to replace another app using ports 3000 or 7860 and shuts down both local processes when its Terminal window closes.

Terminal users can run:

```sh
./setup-local.sh
./start-local.sh
```

The app is available only on `http://127.0.0.1:3000`. The ML engine is available only on `127.0.0.1:7860`, requires a fresh random token for every launch, and is reached through the app's same-origin local bridge. The token is never sent to browser JavaScript.

Finished WAV files and their local index are stored in `data/`. This directory is ignored by Git so private songs—including the local demonstration song—are not accidentally published.

## Model downloads and privacy

The app downloads immutable revisions of:

- `MiniMaxAI/MiniMax-Music3` at `fbdf52fbaaca799592917417eb05f1899f1255ec`
- `mlx-community/whisper-large-v3-turbo` at `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`

Internet access is used only during dependency installation and these model downloads. Generation and lyric synchronization are local. Dependency versions and package hashes are locked in `package-lock.json` and `backend/requirements.txt`.

## Model license and user control

The app does not filter prompts, require an account, or impose an additional usage policy. Users control and may modify their local copy. The model remains governed by the [MiniMax‑Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE), and applicable law still applies.

The UI prominently names MiniMax‑Music3. Anyone distributing a modified or commercial version should review the model's current license directly, including its acceptable-use, safeguards, attribution, and revenue-related conditions. This project does not provide legal advice.

## Open-source notices

- App and MLX implementation: [MIT License](LICENSE)
- Third-party and visual attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- amma.live name and icon: brand identifiers, not granted as trademarks by the MIT software license; forks may replace them
- Security reporting and release threat model: [SECURITY.md](SECURITY.md)
- Contributions: [CONTRIBUTING.md](CONTRIBUTING.md)

## Development and release checks

```sh
npm ci
npm run lint
npm test
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
npm audit
uv pip check --python .venv/bin/python
```

Regenerate the Python lock intentionally with:

```sh
uv pip compile backend/requirements.in --python .venv/bin/python --generate-hashes --output-file backend/requirements.txt
```

## Troubleshooting

- **A local port is already in use:** close the named local app and launch amma.live Music again. The launcher will never kill an unrelated process.
- **The engine is offline:** keep the Terminal window opened by the launcher running, then press **Check engine**.
- **Node is missing:** install Node.js 20.9 or newer from nodejs.org and reopen the app.
- **The Mac runs out of memory:** this release needs 64 GB for supported use. A 48 GB Mac is experimental; close other large apps and start with one minute.
- **Generation looks stuck:** first-time model loading is slow. Progress continues when composition starts.
