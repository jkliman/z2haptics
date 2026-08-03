"""
Profile loading and saving.

The structural tests here exist because of a real bug: `min_flatness` was added
to the Band dataclass and used throughout the engine, but never added to the
loader's allow-list. Every profile silently dropped it at load, so the feature
did nothing -- and only a stray warning line revealed it. A field that exists on
Band must survive a full YAML round trip, and that is now enforced rather than
remembered.
"""

from dataclasses import fields

import pytest
import yaml

from z2haptics.analysis import Band
from z2haptics.profiles import (
    _band_from_dict,
    discover,
    load_profile,
    profile_to_dict,
    save_profile,
)

# Not part of the serialised schema.
NOT_SERIALISED = {"enabled"}


def test_loader_accepts_every_band_field():
    """Every Band field must be loadable from YAML."""
    band_fields = {f.name for f in fields(Band)}
    sample = {f.name: getattr(Band(name="x", low_hz=1, high_hz=2), f.name)
              for f in fields(Band)}

    loaded = _band_from_dict(sample)
    for name in band_fields:
        assert getattr(loaded, name) == sample[name], f"{name} was dropped on load"


def test_serialiser_emits_every_band_field():
    band_fields = {f.name for f in fields(Band)}
    emitted = set(profile_to_dict(
        _profile_with(Band(name="x", low_hz=20, high_hz=90))
    )["bands"][0])
    missing = band_fields - emitted - NOT_SERIALISED
    assert not missing, f"profile_to_dict drops: {sorted(missing)}"


def _profile_with(band: Band):
    from z2haptics.profiles import Profile
    return Profile(name="T", bands=[band])


def test_band_round_trips_through_yaml(tmp_path):
    band = Band(
        name="gunfire", low_hz=90, high_hz=450, sensitivity=2.2, gate=0.0031,
        refractory_ms=110, min_share=0.2, min_flatness=0.45,
        background_subtraction=0.5, max_rate=7, duration_ms=44,
        strength_min=38, strength_max=82, level_floor_db=-50.5,
        level_ceil_db=-24.5, priority=2,
    )
    path = tmp_path / "t.yaml"
    save_profile(_profile_with(band), path)
    back = load_profile(path).bands[0]

    for f in fields(Band):
        if f.name in NOT_SERIALISED:
            continue
        assert getattr(back, f.name) == pytest.approx(getattr(band, f.name)) \
            if isinstance(getattr(band, f.name), float) \
            else getattr(back, f.name) == getattr(band, f.name), \
            f"{f.name} did not survive the round trip"


def test_unknown_keys_are_ignored_not_fatal(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({
        "name": "T",
        "bands": [{"name": "b", "low_hz": 20, "high_hz": 90, "from_the_future": 1}],
    }), encoding="utf-8")
    assert load_profile(path).bands[0].name == "b"


def test_profile_without_bands_is_rejected(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"name": "T", "bands": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(path)


# -- shipped profiles ---------------------------------------------------------

def test_shipped_profiles_load_without_dropping_keys(caplog):
    """A shipped profile using a key the loader rejects is the exact bug above."""
    import logging

    with caplog.at_level(logging.WARNING, logger="z2haptics.profiles"):
        profiles = discover()

    dropped = [r.getMessage() for r in caplog.records if "unknown keys" in r.getMessage()]
    assert not dropped, f"shipped profiles use unrecognised keys: {dropped}"
    assert profiles


def test_shipped_profiles_have_sane_bands():
    for profile in discover().values():
        assert profile.bands, f"{profile.name} has no bands"
        for b in profile.bands:
            assert b.low_hz < b.high_hz, f"{profile.name}/{b.name} inverted range"
            assert 0 <= b.strength_min <= b.strength_max <= 100, \
                f"{profile.name}/{b.name} bad strength range"
            assert b.level_floor_db <= b.level_ceil_db, \
                f"{profile.name}/{b.name} inverted dB window"
            assert b.duration_ms > 0
            assert 0.0 <= b.min_flatness <= 1.0
            assert 0.0 <= b.min_share <= 1.0


def test_fps_profiles_actually_enable_flatness_filtering():
    """The tuning that solves music swamping gunfire must not silently regress."""
    profiles = discover()
    for name in ("FPS", "Battlefield 6"):
        profile = profiles.get(name)
        assert profile is not None, f"{name} profile missing"
        gunfire = next(b for b in profile.bands if b.name == "gunfire")
        assert gunfire.min_flatness >= 0.3, f"{name}: gunfire lost its flatness filter"
        assert gunfire.sensitivity >= 2.0, f"{name}: gunfire sensitivity too low"
