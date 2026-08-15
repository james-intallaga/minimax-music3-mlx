"use client";
/* eslint-disable @next/next/no-img-element */

import type { CSSProperties } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

const API = "/api/local";

type EngineStatus = {
  engine_ready: boolean;
  model_ready: boolean;
  downloading: boolean;
  device: string;
  memory_gb: number;
  model_size_gb: number;
  downloaded_gb: number;
  download_total_gb: number;
  download_progress: number;
  message: string;
};

type Generation = {
  id: string;
  status: "queued" | "loading" | "running" | "completed" | "failed" | "cancelled";
  phase: string;
  progress: number;
  message: string;
  elapsed_seconds: number;
  eta_seconds: number | null;
  audio_url?: string;
  error?: string;
};

type LyricLine = {
  index: number;
  text: string;
  section: boolean;
  start_ms: number;
  end_ms: number;
  detected?: boolean;
};

type Song = {
  id: string;
  title: string;
  caption: string;
  duration_seconds: number;
  created_at: string;
  audio_url: string;
  lyrics?: string;
  lyric_timing?: LyricLine[];
};

type IconName = "play" | "pause" | "back" | "forward" | "shuffle" | "repeat" | "download" | "stop";

const iconPaths: Record<IconName, React.ReactNode> = {
  play: <path d="M8 5v14l11-7z" />,
  pause: <><path d="M7 5h4v14H7z" /><path d="M14 5h4v14h-4z" /></>,
  back: <><path d="M6 5h2v14H6z" /><path d="m18 5-9 7 9 7z" /></>,
  forward: <><path d="M16 5h2v14h-2z" /><path d="m6 5 9 7-9 7z" /></>,
  shuffle: <><path d="M4 7h3c4 0 6 10 10 10h3" fill="none" stroke="currentColor" strokeWidth="2" /><path d="m17 14 3 3-3 3" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M4 17h3c1.5 0 2.7-1.4 3.8-3" fill="none" stroke="currentColor" strokeWidth="2" /></>,
  repeat: <><path d="M17 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M4 11V9a3 3 0 0 1 3-3h13" fill="none" stroke="currentColor" strokeWidth="2" /><path d="m7 21-3-3 3-3" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M20 13v2a3 3 0 0 1-3 3H4" fill="none" stroke="currentColor" strokeWidth="2" /></>,
  download: <><path d="M12 3v12m0 0 5-5m-5 5-5-5M5 21h14" fill="none" stroke="currentColor" strokeWidth="2" /></>,
  stop: <path d="M7 7h10v10H7z" />,
};

function Icon({ name, size = 24 }: { name: IconName; size?: number }) {
  return <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden="true">{iconPaths[name]}</svg>;
}

const durationOptions = [60, 120, 180, 300];
const sampleCaption = "Warm cinematic acoustic pop, 94 BPM, intimate female lead, fingerpicked guitar and soft piano, building into a wide final chorus with strings and layered harmonies.";
const sampleLyrics = `[intro]
[verse]
Morning paints the window gold
A quiet promise we can hold
I hear your footsteps down the hall
And suddenly I fear no fall

[chorus]
Stay with me beneath the moon
We will find our way there soon
Through every dark and every blue
I will keep the light for you

[bridge]
Even when the road is long
Your heartbeat turns the rain to song

[chorus]
Stay with me beneath the moon
We will find our way there soon

[outro]
I will keep the light for you`;

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function formatWait(seconds: number | null) {
  if (!seconds || seconds < 1) return "Estimating…";
  return seconds < 60 ? `${Math.ceil(seconds)} sec left` : `${Math.ceil(seconds / 60)} min left`;
}

function SetupGate({ engine, error, onInstall, onRetry }: { engine: EngineStatus | null; error: string; onInstall: () => void; onRetry: () => void }) {
  const checking = !engine && !error;
  const downloading = !!engine?.downloading;
  const progress = Math.round((engine?.download_progress ?? 0) * 100);

  return (
    <section className="setup-gate" aria-live="polite">
      <div className="setup-mark"><img src="/amma-live-icon.png" alt="" /></div>
      <p className="eyebrow">MiniMax‑Music3 · fully local · open weights</p>
      <h1>{checking ? "Checking your Mac…" : downloading ? "Getting your music studio ready." : error ? "Let’s reconnect the local engine." : "One quick setup, then it’s yours."}</h1>
      <p className="setup-lead">
        {checking && "This only takes a moment."}
        {downloading && "The music model is downloading directly to this Mac. You only do this once."}
        {!checking && !downloading && !error && "Download the music model once. After that, you can make songs privately on this Mac without an account or subscription."}
        {error && "Keep the local app window open, then try again."}
      </p>

      {downloading && (
        <div className="setup-download">
          <div><strong>{progress}%</strong><span>{engine.downloaded_gb.toFixed(1)} of {engine.download_total_gb.toFixed(1)} GB</span></div>
          <div className="install-progress"><i style={{ width: `${Math.max(1, progress)}%` }} /></div>
          <p>You can leave this page open. The composer will appear automatically when the download is finished.</p>
        </div>
      )}

      {!checking && !downloading && !error && <button className="setup-primary" onClick={onInstall}>Download MiniMax‑Music3</button>}
      {error && <><p className="setup-error">{error}</p><button className="setup-primary" onClick={onRetry}>Try again</button></>}

      <div className="setup-notes">
        <span>Apple Silicon</span><span>64 GB supported</span><span>About 29 GB to download</span>
      </div>
    </section>
  );
}

async function readJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.message || "The local engine did not respond.");
  return body as T;
}

function SongPlayer({ song }: { song: Song }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const lineRefs = useRef(new Map<number, HTMLButtonElement>());
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(song.duration_seconds || 0);
  const [playing, setPlaying] = useState(false);
  const [repeat, setRepeat] = useState(false);
  const lines = song.lyric_timing ?? [];
  const currentMs = currentTime * 1000;
  const activeLine = lines.reduce((active, line) => line.start_ms <= currentMs ? line.index : active, -1);
  const progress = duration > 0 ? Math.min(100, currentTime / duration * 100) : 0;

  useEffect(() => {
    if (activeLine < 0) return;
    lineRefs.current.get(activeLine)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeLine]);

  async function toggle() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) await audio.play();
    else audio.pause();
  }

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(duration || song.duration_seconds, seconds));
    setCurrentTime(audio.currentTime);
  }

  return (
    <section className="song-player" aria-label={`Playing ${song.title}`}>
      <audio
        ref={audioRef}
        src={`${API}${song.audio_url}`}
        preload="metadata"
        loop={repeat}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <div className={`vinyl-stage${playing ? " is-playing" : ""}`}>
        <div className={`player-vinyl${playing ? " is-spinning" : ""}`}>
          <img src="/amma-live-icon.png" alt="" />
          <span />
        </div>
        <div className="vinyl-stylus" aria-hidden="true">
          <i className="stylus-pivot" />
          <i className="stylus-arm stylus-arm-one" />
          <i className="stylus-arm stylus-arm-two" />
        </div>
      </div>
      <h2>{song.title}</h2>
      <p className="model-line">MiniMax‑Music3 · open weights · fully local</p>
      <div className="player-progress">
        <span>{formatTime(currentTime)}</span>
        <input
          type="range"
          min="0"
          max={Math.max(1, duration)}
          step="0.01"
          value={Math.min(currentTime, Math.max(1, duration))}
          onChange={(event) => seek(Number(event.currentTarget.value))}
          style={{ "--player-progress": `${progress}%` } as CSSProperties}
          aria-label="Seek song"
        />
        <span>{duration > 0 ? `-${formatTime(Math.max(0, duration - currentTime))}` : "0:00"}</span>
      </div>
      <div className="player-controls">
        <button disabled aria-label="Shuffle"><Icon name="shuffle" size={23} /></button>
        <button onClick={() => seek(currentTime - 10)} aria-label="Back 10 seconds"><Icon name="back" size={29} /></button>
        <button className="main-play" onClick={() => void toggle()} aria-label={playing ? "Pause" : "Play"}>
          <Icon name={playing ? "pause" : "play"} size={36} />
        </button>
        <button onClick={() => seek(currentTime + 10)} aria-label="Forward 10 seconds"><Icon name="forward" size={29} /></button>
        <button className={repeat ? "is-active" : ""} onClick={() => setRepeat((value) => !value)} aria-label="Repeat"><Icon name="repeat" size={23} /></button>
      </div>
      {lines.length > 0 && (
        <div className="synced-lyrics" aria-label="Synced lyrics">
          {lines.map((line) => (
            <button
              key={`${line.index}-${line.text}`}
              ref={(node) => { if (node) lineRefs.current.set(line.index, node); else lineRefs.current.delete(line.index); }}
              className={`${line.section ? "is-section" : ""}${line.index === activeLine ? " is-active" : ""}${line.index < activeLine ? " is-past" : ""}`}
              onClick={() => seek(line.start_ms / 1000)}
            >
              {line.text}
            </button>
          ))}
        </div>
      )}
      <a className="player-download" href={`${API}${song.audio_url}`} download><Icon name="download" size={20} /> Download your song</a>
    </section>
  );
}

export default function Home() {
  const [caption, setCaption] = useState(sampleCaption);
  const [lyrics, setLyrics] = useState(sampleLyrics);
  const [duration, setDuration] = useState(180);
  const [seed, setSeed] = useState(7);
  const [instrumental, setInstrumental] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSongId, setSelectedSongId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const generating = generation && ["queued", "loading", "running"].includes(generation.status);
  const activeSong = useMemo(() => songs.find((song) => song.id === (selectedSongId || generation?.id)) ?? songs[0], [generation?.id, selectedSongId, songs]);
  const modelReady = !!engine?.model_ready;

  async function refreshStatus() {
    try {
      const [nextEngine, nextSongs, current] = await Promise.all([
        readJson<EngineStatus>("/status"),
        readJson<Song[]>("/songs"),
        readJson<Generation | null>("/generations/current"),
      ]);
      setEngine(nextEngine);
      setSongs(nextSongs);
      if (current) {
        setGeneration(current);
        if (current.status === "completed") setSelectedSongId(current.id);
      }
      setError("");
    } catch (cause) {
      setEngine(null);
      setError(cause instanceof Error ? cause.message : "Start the local engine.");
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshStatus(), 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!generating && !engine?.downloading) return;
    const timer = window.setInterval(() => void refreshStatus(), 1200);
    return () => window.clearInterval(timer);
  }, [generating, engine?.downloading]);

  async function createSong() {
    if (!caption.trim()) return setError("Describe your song first.");
    if (!instrumental && !lyrics.trim()) return setError("Add lyrics or choose Instrumental.");
    setBusy(true);
    setError("");
    try {
      const next = await readJson<Generation>("/generations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caption: caption.trim(), lyrics: instrumental ? "[instrumental]" : lyrics.trim(), duration_seconds: duration, seed, steps: 30 }),
      });
      setGeneration(next);
      setSelectedSongId(next.id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start this song.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelSong() {
    try { setGeneration(await readJson<Generation>("/generations/cancel", { method: "POST" })); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not stop yet."); }
  }

  async function downloadModel() {
    setError("");
    try { await readJson("/model/download", { method: "POST" }); await refreshStatus(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not start the model download."); }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="https://amma.live" target="_blank" rel="noreferrer"><img src="/amma-live-icon.png" alt="" /><span>amma.live</span></a>
        <div className={`engine-pill ${modelReady ? "ready" : ""}`}><span className="engine-dot" />{modelReady ? "Local model ready" : engine?.downloading ? "Downloading model" : engine ? "Setup needed" : "Checking Mac"}</div>
      </header>

      {!modelReady ? <SetupGate engine={engine} error={error} onInstall={() => void downloadModel()} onRetry={() => void refreshStatus()} /> : <>
        <section className="hero" id="top">
          <div className="eyebrow">MiniMax‑Music3 · open weights · fully local</div>
          <h1>Freedom To Download Your Song</h1>
          <p>Apple Silicon Mac · 64 GB supported · 48 GB experimental</p>
        </section>

        <section className="workspace">
        <div className="creator-column">
          <div className="form-card">
            <label className="field-label" htmlFor="description"><span>01</span>Describe your song</label>
            <textarea id="description" className="caption-input" value={caption} onChange={(event) => setCaption(event.target.value)} maxLength={4000} />

            <div className="field-row">
              <label className="field-label" htmlFor="lyrics"><span>02</span>Lyrics</label>
              <label className="switch-label"><input type="checkbox" checked={instrumental} onChange={(event) => setInstrumental(event.target.checked)} /><i /><b>Instrumental</b></label>
            </div>
            {!instrumental && <textarea id="lyrics" className="lyrics-input" value={lyrics} onChange={(event) => setLyrics(event.target.value)} />}

            <label className="field-label duration-label"><span>03</span>Length</label>
            <div className="duration-row" role="group" aria-label="Song duration">
              {durationOptions.map((value) => <button key={value} className={duration === value ? "selected" : ""} onClick={() => setDuration(value)}><strong>{value / 60}</strong><small>{value === 60 ? "minute" : "minutes"}</small></button>)}
            </div>

            <button className="advanced-trigger" onClick={() => setAdvancedOpen((open) => !open)} aria-expanded={advancedOpen}><span>Fine tune</span><span>{advancedOpen ? "−" : "+"}</span></button>
            {advancedOpen && <div className="advanced-panel"><label>Seed<input type="number" min={0} max={2147483647} value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label><div><span>Quality</span><strong>30 steps</strong></div><div><span>Output</span><strong>Stereo · 44.1 kHz</strong></div></div>}
            {error && <div className="error-message">{error}</div>}
            <button className="create-button" disabled={busy || !!generating || !engine?.model_ready} onClick={createSong}>{generating ? "Creating your song…" : `Create ${duration / 60}-minute song`}</button>
          </div>
        </div>

        <aside className="result-column">
          {generating ? (
            <section className="generation-view">
              <div className="player-vinyl is-spinning"><img src="/amma-live-icon.png" alt="" /><span /></div>
              <p className="phase">{generation.phase}</p>
              <h2>{generation.message}</h2>
              <div className="generation-progress"><span style={{ width: `${Math.max(2, generation.progress * 100)}%` }} /></div>
              <div className="generation-meta"><span>{Math.round(generation.progress * 100)}%</span><span>{formatWait(generation.eta_seconds)}</span></div>
              <button className="stop-button" onClick={cancelSong}><Icon name="stop" size={20} /> Stop making this song</button>
            </section>
          ) : activeSong ? <SongPlayer key={activeSong.id} song={activeSong} /> : (
            <section className="empty-player"><div className="player-vinyl"><img src="/amma-live-icon.png" alt="" /><span /></div><h2>Your song will appear here.</h2></section>
          )}

          {songs.length > 1 && <div className="song-history"><span>Made here</span>{songs.slice(0, 5).map((song) => <button key={song.id} onClick={() => setSelectedSongId(song.id)} className={activeSong?.id === song.id ? "selected" : ""}><b>{song.title}</b><small>{formatTime(song.duration_seconds)}</small></button>)}</div>}
        </aside>
        </section>

        <section className="how-to" id="how-to">
          <p className="eyebrow">Simple guide</p>
          <h2>How to make your first song</h2>
          <ol>
            <li><span>1</span><div><strong>Describe the sound.</strong><p>Say what kind of music you want, the mood, instruments, tempo, and type of voice.</p></div></li>
            <li><span>2</span><div><strong>Add your words.</strong><p>Paste lyrics, or switch on Instrumental if you do not want singing.</p></div></li>
            <li><span>3</span><div><strong>Choose the length.</strong><p>Try one minute while experimenting. Use three or five minutes for a full song.</p></div></li>
            <li><span>4</span><div><strong>Press Create and leave the page open.</strong><p>Your Mac does all the work. Longer songs take longer, and the progress bar shows what is happening.</p></div></li>
            <li><span>5</span><div><strong>Listen and save it.</strong><p>Lyrics follow the music automatically. Download the finished WAV whenever you are happy.</p></div></li>
          </ol>
          <p className="privacy-note">Your prompt, lyrics, model, and finished songs stay on this computer.</p>
        </section>
      </>}

      <footer><p>MIT open-source app · songs stay local.</p><a className="hosted-option" href="https://amma.live" target="_blank" rel="noreferrer">Mac not supported? Try amma.live</a><button onClick={refreshStatus}>Check engine</button></footer>
    </main>
  );
}
