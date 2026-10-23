import numpy as np
import pytest
from PIL import Image

from partycam.posters import MissingPosters, find_posters

W, H = 1920, 1080


@pytest.fixture
def folder(tmp_path):
    for i in range(3):
        Image.new("RGB", (400, 500), (80 * i, 200, 255 - 80 * i)).save(tmp_path / f"poster{i}.jpg")
    (tmp_path / "notes.txt").write_text("not a poster")
    (tmp_path / ".DS_Store").write_bytes(b"")
    return tmp_path


def run_until_shown(posters: MissingPosters, start: float, quiet: bool = True, step: float = 1.0, limit: float = 3600.0):
    t = start
    while t < start + limit:
        posters.update(t, quiet=quiet)
        if posters.current is not None:
            return t
        t += step
    return None


def test_finds_only_images(folder):
    assert [p.name for p in find_posters(folder)] == ["poster0.jpg", "poster1.jpg", "poster2.jpg"]
    assert find_posters(folder / "missing") == []


def test_first_poster_comes_a_few_minutes_in_then_about_every_ten(folder):
    posters = MissingPosters(folder, every=600, show_for=15, seed=1)
    first = run_until_shown(posters, 0.0)
    assert MissingPosters.FIRST_AFTER[0] <= first <= MissingPosters.FIRST_AFTER[1] + 1
    posters.update(first + 15)
    assert posters.current is None
    second = run_until_shown(posters, first + 15)
    assert 600 * 0.8 <= second - first <= 600 * 1.2


def test_every_poster_shows_once_before_any_repeat(folder):
    posters = MissingPosters(folder, every=60, show_for=5, seed=2)
    seen, t = [], 0.0
    while len(seen) < 3:
        t = run_until_shown(posters, t)
        seen.append(posters.current)
        t += 5
    assert len(set(seen)) == 3


def test_waits_while_something_else_is_on_screen(folder):
    posters = MissingPosters(folder, seed=3)
    posters.update(0.0)
    posters.update(posters.next_at + 1, quiet=False)
    assert posters.current is None
    posters.update(posters.next_at + 2, quiet=True)
    assert posters.current is not None


def test_busy_holds_others_back_briefly_after(folder):
    posters = MissingPosters(folder, show_for=15, seed=4)
    posters.trigger()
    posters.update(100.0)
    assert posters.busy(100.0)
    posters.update(115.0)
    assert posters.current is None
    assert posters.busy(115.0 + MissingPosters.COOLDOWN - 0.1)
    assert not posters.busy(115.0 + MissingPosters.COOLDOWN)


def test_trigger_works_even_when_off_and_skips_to_next(folder):
    posters = MissingPosters(folder, enabled=False, seed=5)
    assert run_until_shown(posters, 0.0, limit=1200) is None
    posters.trigger()
    posters.update(1300.0)
    first = posters.current
    posters.trigger()
    posters.update(1301.0)
    assert first is not None and posters.current not in (None, first)


def test_no_posters_never_shows(tmp_path):
    posters = MissingPosters(tmp_path / "missing")
    posters.trigger()
    assert run_until_shown(posters, 0.0) is None
    assert not posters.busy(10_000.0)


def test_draw_replaces_frame_and_cuts_back(folder):
    posters = MissingPosters(folder, show_for=15, seed=6)
    live = np.full((H, W, 3), 40, np.uint8)
    frame = live.copy()
    posters.draw(frame, 0.0)
    assert np.array_equal(frame, live)  # nothing up

    posters.trigger()
    posters.update(0.0)
    frame = live.copy()
    posters.draw(frame, 0.0)
    assert np.array_equal(frame, live)  # first instant of the cut-in is still the live frame
    held = live.copy()
    posters.draw(held, 7.0)
    assert held.shape == live.shape
    # Red alert bar on top, poster colors in the middle, dark caption strip at the bottom
    assert tuple(held[5, 5]) == (28, 22, 196)
    assert tuple(held[H // 2, W // 2]) != (40, 40, 40)
    assert held[H - 5, 5].max() < 30
    # Slow push-in: the poster grows over the showing
    later = live.copy()
    posters.draw(later, 14.0)
    assert not np.array_equal(held, later)
