import unittest
import numpy as np

from bitemporal_present import BitemporalPresent


class BitemporalPresentTests(unittest.TestCase):
    def test_late_observation_has_old_world_time_and_new_knowledge_time(self):
        p = BitemporalPresent(dim=1, dt=0.1, sources=["camera"])
        p.advance(10)
        p.observe(np.array([7.0]), world_tick=4, source="camera")
        snap = p.project(past_ticks=8, future_ticks=2)
        row = list(snap.relative_ticks).index(-6)
        self.assertEqual(snap.observation_mask[row], 1.0)
        self.assertEqual(snap.observation_value[row, 0], 7.0)
        self.assertEqual(snap.observation_knowledge_age[row], 0.0)

    def test_prediction_can_live_in_future_then_become_due(self):
        p = BitemporalPresent(dim=1, dt=0.1)
        p.predict(np.array([2.0]), world_tick=3)
        self.assertEqual(len(p.due_predictions()), 0)
        p.advance(3)
        due = p.due_predictions()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].value[0], 2.0)

    def test_watermarks_distinguish_missing_from_complete_empty(self):
        p = BitemporalPresent(dim=1, dt=0.1, sources=["camera", "mic"])
        p.advance(10)
        p.set_watermark("camera", 8)
        self.assertEqual(p.completeness_at(8), 0.5)
        self.assertEqual(p.completeness_at(9), 0.0)
        p.set_watermark("mic", 8)
        self.assertEqual(p.completeness_at(8), 1.0)

    def test_future_rows_never_claim_evidence_complete(self):
        p = BitemporalPresent(dim=1, dt=0.1, sources=["camera"])
        p.advance(5)
        p.set_watermark("camera", 5)
        snap = p.project(past_ticks=0, future_ticks=2)
        self.assertEqual(snap.completeness[0], 1.0)
        self.assertEqual(snap.completeness[1], 0.0)
        self.assertEqual(snap.completeness[2], 0.0)

    def test_watermark_cannot_move_backwards_or_beyond_now(self):
        p = BitemporalPresent(dim=1, dt=0.1, sources=["camera"])
        p.advance(5)
        p.set_watermark("camera", 3)
        with self.assertRaises(ValueError):
            p.set_watermark("camera", 2)
        with self.assertRaises(ValueError):
            p.set_watermark("camera", 6)


if __name__ == "__main__":
    unittest.main()
