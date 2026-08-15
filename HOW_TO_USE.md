# Make music on your Mac with amma.live

This app makes music on your own Apple Silicon Mac. Your description, lyrics, model, and finished audio stay on your computer. You do not need an amma.live account, subscription, API key, or cloud storage.

## Before you begin

You need a Mac with Apple Silicon and **64 GB of memory** for supported use. A 48 GB Mac is experimental. The current version does not support 32–36 GB Macs. Keep about 42 GB of storage free and connect to the internet for the first setup.

## Open the app

Double-click **Open amma.live Music.command**. The first launch prepares the app, which can take several minutes. Keep the Terminal window open; closing it safely stops the private local music engine.

If another local app is already using the required address, amma.live Music tells you which app to close instead of opening the wrong page.

## Download the model once

The app checks whether MiniMax‑Music3 is on your Mac. If it is ready, the composer opens immediately. Otherwise:

1. Click **Download MiniMax‑Music3**.
2. Leave the page open while the approximately 29 GB download progresses.

The composer appears automatically when the download is complete. You do not repeat this next time.

## Make a song

1. **Describe the sound.** Say the genre, mood, tempo, instruments, kind of voice, and how the music should develop.
2. **Add your own lyrics.** Put labels such as `[verse]`, `[chorus]`, and `[bridge]` on separate lines. Choose **Instrumental** for music without singing.
3. **Choose the length.** Try one minute first. Choose three or five minutes for a complete song.
4. **Press Create.** The progress bar explains what the Mac is doing. Longer songs can take a long time.
5. **Listen and save.** The lyrics follow the song automatically. Select **Download WAV** to save another copy.

Example description:

> Warm cinematic acoustic pop, 94 BPM, intimate female lead, fingerpicked guitar and soft piano, building into a wide final chorus with strings and layered harmonies.

## Stop safely

Select **Stop making this song** during generation. The engine stops after the current safe processing point and keeps previously finished songs.

## Your local copy

Finished songs are stored in the project's `data/songs` folder. This folder is private by default and is not included when publishing the code.

The app does not filter your prompts or decide what you can create. You control the local copy and can modify its open-source code. The MiniMax‑Music3 Community License and applicable law still apply independently of the app.

## If something looks wrong

- **Engine offline:** keep the Terminal window open, then select **Try again** or **Check engine**.
- **Not enough memory:** 64 GB is supported; 48 GB is experimental. Close large apps and try one minute.
- **No sound yet:** audio appears after the WAV is completely produced and saved.
- **Very slow progress:** local generation is demanding. A five-minute song can take tens of minutes.
