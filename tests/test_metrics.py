import unittest
import sys
import os

# Ensure the src directory is on the path to import src.metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import calculate_ctl_atl

class TestMetrics(unittest.TestCase):
    def test_calculate_ctl_atl_empty(self):
        ctl, atl = calculate_ctl_atl([])
        self.assertEqual(ctl, [])
        self.assertEqual(atl, [])

    def test_calculate_ctl_atl_constant_load(self):
        # A constant load of 100 for 1000 days
        loads = [100.0] * 1000
        ctl, atl = calculate_ctl_atl(loads)
        
        self.assertEqual(len(ctl), 1000)
        self.assertEqual(len(atl), 1000)

        # After 1000 days, EWMA should approach 100
        self.assertAlmostEqual(ctl[-1], 100.0, places=1)
        self.assertAlmostEqual(atl[-1], 100.0, places=1)

    def test_calculate_ctl_atl_zero_load(self):
        # A constant 0 load
        loads = [0.0] * 10
        ctl, atl = calculate_ctl_atl(loads)
        
        self.assertEqual(ctl[-1], 0)
        self.assertEqual(atl[-1], 0)

    def test_calculate_ctl_atl_decay_behavior(self):
        # 1 day of high load, then 0s
        loads = [100.0] + [0.0] * 20
        ctl, atl = calculate_ctl_atl(loads)
        
        # Day 1:
        # CTL: 0 + 100/42 = 2.38
        # ATL: 0 + 100/7 = 14.28
        self.assertAlmostEqual(ctl[0], 100/42, places=2)
        self.assertAlmostEqual(atl[0], 100/7, places=2)
        
        # ATL (decay 7) should decay generally faster than CTL (decay 42)
        # By the 21st day (index 20), ATL should be noticeably lower than it was
        # CTL = 2.38 * (41/42)^20
        # ATL = 14.28 * (6/7)^20
        expected_ctl = (100/42) * ((41.0/42.0)**20)
        expected_atl = (100/7) * ((6.0/7.0)**20)
        
        self.assertAlmostEqual(ctl[-1], expected_ctl, places=2)
        self.assertAlmostEqual(atl[-1], expected_atl, places=2)
        self.assertTrue(atl[-1] < ctl[-1], "ATL should decay below CTL over time")

if __name__ == '__main__':
    unittest.main()
