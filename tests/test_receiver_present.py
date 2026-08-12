import unittest

from receiver_present import ReceiverPresent


class ReceiverPresentTests(unittest.TestCase):
    def test_same_world_age_different_receiver_maturity(self):
        present = ReceiverPresent(
            {
                ("sensor", "fast"): 2,
                ("sensor", "slow"): 7,
            }
        )
        present.advance(10)
        present.emit("sensor", 1.0)
        present.advance(3)

        fast = present.snapshot("fast")
        slow = present.snapshot("slow")

        self.assertEqual(present.now_tick, 13)
        self.assertEqual(fast.path_frontiers["sensor"], 11)
        self.assertEqual(slow.path_frontiers["sensor"], 6)
        self.assertEqual(present.frontier_width("sensor"), 5)

        fast_item = fast.latest_arrived("sensor")
        self.assertIsNotNone(fast_item)
        assert fast_item is not None
        self.assertEqual(fast_item.world_age(present.now_tick), 3)
        self.assertEqual(fast_item.arrival_age(present.now_tick), 1)

        self.assertIsNone(slow.latest_arrived("sensor"))
        self.assertEqual(len(slow.in_flight), 1)
        slow_item = slow.in_flight[0]
        self.assertEqual(slow_item.world_age(present.now_tick), 3)
        self.assertGreater(slow_item.path_progress(present.now_tick), 0.0)
        self.assertLess(slow_item.path_progress(present.now_tick), 1.0)

    def test_zero_delay_arrives_immediately(self):
        present = ReceiverPresent({("clock", "reader"): 0})
        present.advance(4)
        present.emit("clock", "tick")
        snap = present.snapshot("reader")
        item = snap.latest_arrived("clock")
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.arrival_tick, 4)
        self.assertEqual(item.path_progress(4), 1.0)

    def test_future_world_event_rejected(self):
        present = ReceiverPresent({("sensor", "reader"): 1})
        with self.assertRaises(ValueError):
            present.emit("sensor", 1.0, world_tick=1)


if __name__ == "__main__":
    unittest.main()
