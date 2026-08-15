# Security policy

## Supported version

Security fixes are applied to the latest version on the default branch.

## Report a vulnerability

Please do not publish an exploitable vulnerability before a fix is available. Report it privately to the repository owner through GitHub's private vulnerability reporting feature. Include the affected version, reproduction steps, impact, and any suggested mitigation. Do not include private songs, model files, access tokens, or other personal data.

## Local threat model

This project is designed for one person on one Mac. Both services bind only to `127.0.0.1`. Each launch creates a new random engine token, browser JavaScript never receives it, and the same-origin bridge rejects browser requests from other websites. The app has no account system and should not be exposed to a LAN, public IP address, reverse proxy, or hosted multi-user environment without a separate security design.

Generated songs and prompts are ordinary local files under `data/`. Anyone with access to the user's macOS account may be able to read them. Full-disk encryption and normal macOS account security remain the user's responsibility.
