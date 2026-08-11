import unittest

from temporal_kernel import (
    derive_temporal_state,
    human_duration,
    is_stale,
    parse_time,
    select_variant_time,
)


class TemporalKernelTests(unittest.TestCase):
    def test_variant_selection_and_ages(self):
        history = [
            {"role": "system", "content": "s", "time": "2026-01-01T00:00:00Z"},
            {"role": "tool", "name": "sensor", "content": "x", "time": "2026-01-01T00:00:10Z"},
            {
                "role": "user",
                "content": "next",
                "time": [
                    "2026-01-01T00:00:20Z",
                    "2026-01-01T01:00:10Z",
                    "2026-01-03T00:00:10Z",
                ],
            },
        ]
        state = derive_temporal_state(history, level=1)
        self.assertEqual(state.conversation_age_seconds, 3610.0)
        self.assertEqual(state.last_tool_age_seconds, 3600.0)
        self.assertEqual(state.messages[-1].age_seconds, 0.0)

    def test_state_render_does_not_make_freshness_decision(self):
        history = [
            {"role": "system", "content": "s", "time": "2026-01-01T00:00:00Z"},
            {"role": "tool", "content": "x", "time": "2026-01-01T00:00:01Z"},
            {"role": "user", "content": "u", "time": "2026-01-02T00:00:00Z"},
        ]
        text = derive_temporal_state(history).render().lower()
        self.assertNotIn("stale", text)
        self.assertNotIn("reuse", text)
        self.assertNotIn("call tool", text)

    def test_clock_anomaly_is_not_silently_clamped(self):
        history = [
            {"role": "user", "content": "future", "time": "2026-01-02T00:00:00Z"},
            {"role": "user", "content": "now", "time": "2026-01-01T00:00:00Z"},
        ]
        with self.assertRaises(ValueError):
            derive_temporal_state(history)

    def test_parse_z_and_offset_are_same_instant(self):
        a = parse_time("2026-01-01T00:00:00Z")
        b = parse_time("2026-01-01T02:00:00+02:00")
        self.assertEqual(a, b)

    def test_helpers(self):
        self.assertEqual(select_variant_time(["a", "b", "c"], 2), "c")
        self.assertEqual(human_duration(90061), "1d 1h 1m 1s")
        self.assertFalse(is_stale(10, 10))
        self.assertTrue(is_stale(10.001, 10))


if __name__ == "__main__":
    unittest.main()
