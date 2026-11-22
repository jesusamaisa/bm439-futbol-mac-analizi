"""Homografi: saha anahtar noktalarının FIFA ölçülerindeki konumları."""
from __future__ import annotations

import pytest

from bgoru.homography import KP_METERS, KP_NAMES


def test_pitch_keypoints_are_symmetric_and_in_bounds():
    assert len(KP_NAMES) == 28 and set(KP_NAMES) == set(KP_METERS)
    for name, (x, y) in KP_METERS.items():
        assert 0 <= x <= 105 and 0 <= y <= 68, name
        mirror = name.replace("T", "#").replace("B", "T").replace("#", "B") if name[0] in "TB" else None
        if mirror and mirror in KP_METERS:  # üst/alt (sol/sağ kale) uçları ayna simetrik
            assert KP_METERS[mirror][0] == pytest.approx(105 - x) and KP_METERS[mirror][1] == pytest.approx(y)
    assert KP_METERS["TL18MC"][0] == pytest.approx(16.5)          # ceza sahası derinliği
    w = KP_METERS["TR18ML"][1] - KP_METERS["TL18ML"][1]
    assert w == pytest.approx(40.32)                              # ceza sahası genişliği
