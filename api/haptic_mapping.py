"""
Map stress probability to a haptic pattern.

This is the "generation" side of the emotion-transfer pipeline.

Design note (IMPORTANT)
-----------------------
There is no established scientific literature showing that a given
vibration pattern reliably induces a specific affective state in a
receiver. Our mapping is a *proposed* correspondence, not a
validated intervention. It should be treated as a design prototype,
not as a proven induction mechanism.

For the demo we use a simple three-band mapping:

    Low stress    (p < 0.35)  → gentle, slow pattern
    Medium stress (0.35-0.65) → moderate, rhythmic pattern
    High stress   (p > 0.65)  → strong, fast pattern

Each pattern specifies:
    intensity  (0-1)   vibration strength
    frequency  (Hz)    pulses per second
    duration_ms        total pattern length
    pulses             list of {t_ms, intensity} events

The React client can render these either as CSS animations
(simulation) or as Web Vibration API calls (on mobile).
"""

from typing import Dict, List, Any


def _build_pulses(n_pulses: int,
                  spacing_ms: int,
                  intensity: float,
                  decay: float = 1.0) -> List[Dict[str, float]]:
    """
    Build a list of vibration pulses.

    Parameters
    ----------
    n_pulses : int
    spacing_ms : int
        Time between pulse onsets.
    intensity : float
        Base intensity (0-1).
    decay : float
        Multiplicative decay per pulse (1.0 = no decay).
    """
    pulses = []
    current_intensity = intensity
    for i in range(n_pulses):
        pulses.append({
            "t_ms": i * spacing_ms,
            "intensity": round(current_intensity, 3),
        })
        current_intensity *= decay
    return pulses


def stress_to_haptic(stress_probability: float,
                     style: str = "continuous") -> Dict[str, Any]:
    """
    Map a stress probability to a haptic pattern.

    Parameters
    ----------
    stress_probability : float
        Between 0 and 1.
    style : str
        "continuous" (default): pattern length scales with probability.
        "banded": three discrete bands (gentle/moderate/strong).

    Returns
    -------
    dict with keys:
        band         : "low" | "medium" | "high"
        intensity    : float 0-1 (overall)
        frequency_hz : pulses per second
        duration_ms  : total pattern length
        pulses       : list of individual pulses
        description  : human-readable summary
    """

    p = max(0.0, min(1.0, float(stress_probability)))

    # --- Three-band classification ---
    if p < 0.35:
        band = "low"
        base_intensity = 0.2 + 0.3 * p          # 0.20 – 0.31
        spacing_ms = 500                         # 2 Hz
        n_pulses = 4
        decay = 0.9
        description = "Gentle, slow pulses"
    elif p < 0.65:
        band = "medium"
        base_intensity = 0.5 + 0.3 * (p - 0.35) / 0.30  # 0.50 – 0.80
        spacing_ms = 300                         # ~3.3 Hz
        n_pulses = 6
        decay = 0.95
        description = "Moderate, rhythmic pulses"
    else:
        band = "high"
        base_intensity = 0.75 + 0.25 * (p - 0.65) / 0.35  # 0.75 – 1.00
        spacing_ms = 150                         # ~6.7 Hz
        n_pulses = 8
        decay = 1.0
        description = "Strong, fast pulses"

    pulses = _build_pulses(n_pulses, spacing_ms, base_intensity, decay)
    duration_ms = (n_pulses - 1) * spacing_ms + 200  # last pulse + 200ms
    frequency_hz = 1000.0 / spacing_ms

    return {
        "band": band,
        "intensity": round(base_intensity, 3),
        "frequency_hz": round(frequency_hz, 2),
        "duration_ms": duration_ms,
        "pulses": pulses,
        "description": description,
        "stress_probability": round(p, 4),
    }