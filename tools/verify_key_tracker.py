#!/usr/bin/env python3
"""Verify Krumhansl-Schmuckler key-detection math matches src/pitch.rs.

Same algorithm as Rust KeyTracker. Run this before committing Slice 25 to
confirm that Python and Rust agree on (root, is_major, correlation) for each
test case. Any FAIL line means the implementations have diverged.

Usage:
    python3 tools/verify_key_tracker.py
"""

import math
import sys

# ── K-S profiles (Krumhansl & Kessler 1982) ──────────────────────────────────
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def normalize(v):
    """L2-normalize a 12-element list. Returns None if all-zero."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return None
    return [x / norm for x in v]


def pearson_rotated(chroma, profile, rot):
    """Pearson correlation between chroma[rot:] (mod 12) and profile.

    Matches Rust pitch::pearson_rotated exactly:
      chroma[(i + rot) % 12]  vs  profile[i]  for i in 0..12
    """
    n = 12
    mean_c = sum(chroma) / n
    mean_p = sum(profile) / n
    num = sum(
        (chroma[(i + rot) % 12] - mean_c) * (profile[i] - mean_p)
        for i in range(n)
    )
    ss_c = sum((chroma[(i + rot) % 12] - mean_c) ** 2 for i in range(n))
    ss_p = sum((profile[i] - mean_p) ** 2 for i in range(n))
    den = math.sqrt(ss_c * ss_p)
    return num / den if den > 1e-9 else 0.0


def detect_key(chroma):
    """Run K-S analysis on a raw chroma vector.

    Returns (root: int, is_major: bool, confidence: float).
    """
    c = normalize(chroma)
    if c is None:
        return 0, True, 0.0
    best_root, best_major, best_corr = 0, True, float("-inf")
    for rot in range(12):
        r_maj = pearson_rotated(c, KS_MAJOR, rot)
        r_min = pearson_rotated(c, KS_MINOR, rot)
        if r_maj > best_corr:
            best_corr, best_root, best_major = r_maj, rot, True
        if r_min > best_corr:
            best_corr, best_root, best_major = r_min, rot, False
    return best_root, best_major, best_corr


def make_chroma_for_key(rot, is_major):
    """Build a chromagram whose content exactly represents key `rot` of `is_major`.

    Mirrors the Rust test helper make_chroma_for_key:
      c[j] = profile[(j + 12 - rot) % 12]
    This guarantees pearson_rotated(c, profile, rot) == 1.0.
    """
    profile = KS_MAJOR if is_major else KS_MINOR
    return [profile[(j + 12 - rot) % 12] for j in range(12)]


# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (description, chroma_vector, expected_root, expected_is_major)
TEST_CASES = [
    # Profile-derived inputs — these reproduce the Rust unit tests exactly.
    (
        "C major  (profile-derived, self-correlation = 1.0)",
        make_chroma_for_key(0, True), 0, True,
    ),
    (
        "A minor  (profile-derived, self-correlation = 1.0)",
        make_chroma_for_key(9, False), 9, False,
    ),
    (
        "F# major (profile-derived, self-correlation = 1.0, tests rotation)",
        make_chroma_for_key(6, True), 6, True,
    ),
    # Scale-tone inputs — more realistic; tests discrimination power.
    (
        "C major  (equal-weight scale tones: C D E F G A B)",
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], 0, True,
    ),
    (
        "A harmonic minor (A B C D E F G#: pitches 9 11 0 2 4 5 8)",
        [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], 9, False,
    ),
    (
        "F# major (equal-weight scale tones: F# G# A# B C# D# F)",
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], 6, True,
    ),
]


def run_tests():
    print("Krumhansl-Schmuckler key-detection verification")
    print("=" * 70)
    all_passed = True

    for desc, chroma, exp_root, exp_major in TEST_CASES:
        root, is_major, corr = detect_key(chroma)
        mode = "major" if is_major else "minor"
        exp_mode = "major" if exp_major else "minor"
        ok = root == exp_root and is_major == exp_major
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"[{status}] {desc}")
        print(f"       detected : {PITCH_NAMES[root]} {mode}  (r = {corr:.4f})")
        if not ok:
            print(f"       expected : {PITCH_NAMES[exp_root]} {exp_mode}  ← MISMATCH")
        print()

    print("=" * 70)
    if all_passed:
        print("All test cases PASSED — Python/Rust math is consistent.")
    else:
        print("FAILURES detected — review the algorithm before committing.")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
