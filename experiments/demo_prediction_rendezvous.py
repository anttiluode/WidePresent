"""Mechanical demo: a prediction is scheduled in the future and later meets now."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from widepresent import WidePresent

wp = WidePresent(dim=1, dt=0.1, past_ticks=4, future_ticks=5)
wp.schedule(np.array([1.0]), ticks_ahead=3)

for i in range(5):
    observation = np.array([1.0]) if i == 2 else np.array([0.0])
    due = wp.tick(observation)
    err = observation - due
    print(
        f"tick={i:02d} t={i*wp.dt:0.1f}s "
        f"due={due[0]:.1f} obs={observation[0]:.1f} error={err[0]:+.1f}"
    )
