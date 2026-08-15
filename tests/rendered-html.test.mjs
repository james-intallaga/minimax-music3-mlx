import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);

test("ships the complete local music experience", async () => {
  const [page, layout, css, backend] = await Promise.all([
    readFile(new URL("app/page.tsx", projectRoot), "utf8"),
    readFile(new URL("app/layout.tsx", projectRoot), "utf8"),
    readFile(new URL("app/globals.css", projectRoot), "utf8"),
    readFile(new URL("backend/main.py", projectRoot), "utf8"),
  ]);

  assert.match(layout, /MiniMax Music 3 · MLX/);
  assert.match(page, /Freedom To Download Your Song/);
  assert.match(page, /durationOptions = \[60, 120, 180, 300\]/);
  assert.match(page, /Download WAV/);
  assert.match(page, /MiniMax‑Music3 · open weights · fully local/);
  assert.match(page, /onTimeUpdate/);
  assert.match(page, /synced-lyrics/);
  assert.match(css, /@media \(max-width: 600px\)/);
  assert.match(css, /@media \(max-width: 900px\)/);
  assert.match(css, /player-vinyl\.is-spinning/);
  assert.match(backend, /duration_seconds: int = Field\(ge=60, le=300\)/);
  assert.match(backend, /require_local_token/);
  assert.match(backend, /MUSIC_MODEL_REVISION/);
  assert.match(backend, /mlx_whisper\.transcribe/);
  assert.match(backend, /word_timestamps=True/);
});

test("keeps amma.live separate from the local app", async () => {
  const page = await readFile(new URL("app/page.tsx", projectRoot), "utf8");
  assert.doesNotMatch(page, /api\.amma\.live|stripe|analytics|supabase/i);
  assert.match(page, />amma\.live<\/span>/);
  assert.match(page, /Mac not supported\? Try amma\.live/);
  assert.match(page, /API = "\/api\/local"/);
});
