# Contributing

Contributions are welcome when they preserve the project's local-first privacy model and the licenses of all included materials.

Before opening a pull request:

1. Do not commit model weights, generated songs, prompts, secrets, personal paths, or user data.
2. Add attribution and license text for copied or adapted code, images, audio, or other assets.
3. Keep the engine and web app bound to loopback addresses unless a reviewed security design explicitly changes the threat model.
4. Run `npm ci`, `npm run lint`, `npm test`, the Python unit tests, `npm audit`, and a Python dependency audit.
5. Explain hardware used for inference changes and do not lower published memory requirements without a successful end-to-end benchmark.

By contributing, you agree that your contribution may be distributed under the repository's MIT License and that you have the right to submit it.
