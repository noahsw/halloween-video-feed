import numpy as np
import pytest

from partycam.zones import IgnoreZones, in_any_zone

# A zone drawn around a skeleton DJ prop, in 1920x1080 pixels
DJ_ZONE = [np.array([564.0, 206.0, 709.0, 551.0])]


@pytest.mark.parametrize(
    "box, ignored",
    [
        ([574, 217, 687, 394], True),  # the model boxed only the DJ's upper body (an IoU rule missed this)
        ([572, 214, 701, 543], True),  # the whole DJ
        ([520, 300, 760, 900], False),  # a person dancing in front of the DJ: closer, so bigger
        ([900, 400, 1100, 900], False),  # someone elsewhere on the floor
        ([600, 230, 630, 280], False),  # something tiny inside the zone (not the prop)
        ([600, 300, 600, 400], False),  # zero-area box
    ],
)
def test_dj_zone_rule(box, ignored):
    assert in_any_zone(np.array(box, float), DJ_ZONE) is ignored


def test_no_zones_ignores_nothing():
    assert not in_any_zone(np.array([0.0, 0.0, 100.0, 100.0]), [])


def test_zones_are_saved_resolution_independent(tmp_path):
    path = tmp_path / "zones.json"
    zones = IgnoreZones(path)
    zones.add(564, 206, 145, 345, 1920, 1080)

    reloaded = IgnoreZones(path)
    assert len(reloaded.zones) == 1
    # Same zone at half resolution is half the size
    x1, y1, x2, y2 = reloaded.pixel_boxes(960, 540)[0]
    assert (x1, y1, x2, y2) == pytest.approx((282, 103, 354.5, 275.5))

    reloaded.clear()
    assert IgnoreZones(path).zones == []
