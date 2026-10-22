import random
from collections import Counter
from datetime import datetime

import numpy as np
import pytest

from partycam import messages
from partycam.banner import JoinBanner, ShuffleBag, _clock_pools, _clock_text, _floor_pool

POOLS = {name: getattr(messages, name) for name in dir(messages) if name.isupper()}


def test_messages_are_plain_unique_and_fit_on_screen():
    everything = [m for pool in POOLS.values() for m in pool]
    assert len(everything) == len(set(everything))
    assert all(m.isascii() for m in everything)  # the banner font has no fancy glyphs
    assert all(len(m) <= 50 for m in everything)


def test_live_templates_fill_in():
    for m in messages.LIVE_COUNT + messages.LIVE_TIME:
        filled = m.format(n=22, next=23, time="11:42 PM")
        assert "{" not in filled


def test_shuffle_bag_shows_everything_once_per_round():
    items = [f"m{i}" for i in range(30)]
    bag = ShuffleBag(items, random.Random(0))
    assert sorted(bag.next() for _ in range(30)) == sorted(items)


def test_shuffle_bag_holds_back_recent_messages_across_reshuffles():
    items = [f"m{i}" for i in range(30)]
    bag = ShuffleBag(items, random.Random(1))
    shown = [bag.next() for _ in range(30 * 20)]
    assert all(a != b for a, b in zip(shown, shown[1:]))
    held = len(items) // 3
    for r in range(1, 20):
        end_of_last_round = set(shown[r * 30 - held : r * 30])
        start_of_this_round = shown[r * 30 : r * 30 + (30 - held)]
        assert not end_of_last_round & set(start_of_this_round)


@pytest.mark.parametrize("count, pool", [(0, "EMPTY"), (1, "SOLO"), (3, "FEW"), (12, "CROWD"), (20, "PACKED"), (40, "PACKED")])
def test_floor_busyness_tiers(count, pool):
    assert _floor_pool(count) == pool


@pytest.mark.parametrize(
    "hh, mm, expected",
    [
        (20, 15, {"EARLY"}),
        (22, 30, set()),
        (23, 50, {"WITCHING"}),
        (0, 10, {"WITCHING"}),
        (1, 20, {"LATE"}),
        (1, 50, {"LATE", "LAST_CALL"}),
        (14, 0, set()),
    ],
)
def test_time_of_night_pools(hh, mm, expected):
    assert set(_clock_pools(datetime(2026, 10, 31, hh, mm))) == expected


@pytest.mark.parametrize("hh, mm, text", [(0, 5, "12:05 AM"), (9, 30, "9:30 AM"), (12, 0, "12:00 PM"), (23, 42, "11:42 PM")])
def test_clock_text(hh, mm, text):
    assert _clock_text(datetime(2026, 10, 31, hh, mm)) == text


def simulate(hours, count, start=datetime(2026, 10, 31, 20, 0), seed=3):
    """Runs the banner's scheduling like the app does (no drawing). Returns [(t, message)]."""
    b = JoinBanner(every=30.0, seed=seed)
    t0 = start.timestamp()
    shown, last, t = [], None, t0
    while t < t0 + hours * 3600:
        b.counts.append((t, count))
        while b.counts and b.counts[0][0] < t - b.SITUATION_WINDOW:
            b.counts.popleft()
        b._schedule(t)
        if b.current is not None and b.current is not last:
            shown.append((t, b.current))
        last = b.current
        t += 0.5
    return shown


def test_messages_come_every_half_minute_without_back_to_back_repeats():
    shown = simulate(hours=2, count=25)
    gaps = np.diff([t for t, _ in shown])
    assert 20 <= gaps.min() and gaps.max() <= 40
    texts = [m for _, m in shown]
    assert all(a != b for a, b in zip(texts, texts[1:]))
    first_repeat = next(t for i, (t, m) in enumerate(shown) if m in texts[:i])
    assert first_repeat - shown[0][0] > 20 * 60  # nothing repeats in the first 20 minutes


def test_situational_pools_respect_their_cooldowns():
    shown = simulate(hours=2, count=25)
    packed = [t for t, m in shown if m in messages.PACKED]
    assert packed, "a packed floor should get packed-floor messages"
    assert np.diff(packed).min() >= JoinBanner.COOLDOWNS["PACKED"]


def test_live_count_shows_the_real_number():
    shown = simulate(hours=1, count=22)
    live = [m for _, m in shown if "SKELETON" in m and any(c.isdigit() for c in m) and ":" not in m]
    assert live and all("22" in m or "23" in m for m in live)


def test_mostly_general_messages():
    shown = simulate(hours=3, count=25)
    general = Counter(m in messages.GENERAL for _, m in shown)
    assert general[True] / len(shown) > 0.6
