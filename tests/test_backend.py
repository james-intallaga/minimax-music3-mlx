import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import main
from backend.main import GenerationRequest, fallback_lyric_timing, normalized_words


class LocalEngineTests(unittest.TestCase):
    def test_supports_five_minute_requests(self):
        request = GenerationRequest(
            caption="warm acoustic pop",
            lyrics="[instrumental]",
            duration_seconds=300,
        )
        self.assertEqual(request.duration_seconds, 300)

    def test_rejects_more_than_five_minutes(self):
        with self.assertRaises(ValidationError):
            GenerationRequest(
                caption="warm acoustic pop",
                lyrics="[instrumental]",
                duration_seconds=301,
            )

    def test_fallback_lyric_timing_is_monotonic(self):
        lines = fallback_lyric_timing(
            "[verse]\nMorning paints the window gold\nA quiet promise we can hold",
            30.0,
        )
        starts = [line["start_ms"] for line in lines]
        self.assertEqual(starts, sorted(starts))
        self.assertLessEqual(lines[-1]["end_ms"], 30_000)

    def test_word_normalization_handles_curly_apostrophes(self):
        self.assertEqual(normalized_words("We’ll stay"), ["we'll", "stay"])

    def test_engine_rejects_requests_without_launch_token(self):
        original = main.LOCAL_API_TOKEN
        main.LOCAL_API_TOKEN = "test-secret"
        try:
            response = TestClient(main.app).get("/api/status")
            self.assertEqual(response.status_code, 401)
        finally:
            main.LOCAL_API_TOKEN = original

    def test_engine_accepts_the_current_launch_token(self):
        original = main.LOCAL_API_TOKEN
        main.LOCAL_API_TOKEN = "test-secret"
        try:
            response = TestClient(main.app).get(
                "/api/status",
                headers={"Authorization": "Bearer test-secret"},
            )
            self.assertEqual(response.status_code, 200)
        finally:
            main.LOCAL_API_TOKEN = original

if __name__ == "__main__":
    unittest.main()
