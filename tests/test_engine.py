"""Tests of the calculation, which runs the real SECS 2R engine."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import secs2r

from secs_room_correction.engine import (
    ChannelFilters,
    CorrectionError,
    PeqCorrection,
    calculate,
    calculate_peq,
)
from tests import FIXTURES


def test_stereo_project_gives_a_stereo_impulse_response(tmp_path: Path) -> None:
    """A 2.0 project becomes one stereo float impulse response with the gain afir takes away."""
    project = _copy_fixture(tmp_path)
    correction = calculate(project)

    assert correction.sample_rate == 48000
    assert correction.design_id
    samples = _float_samples(correction.wav)
    assert samples.shape[1] == 2
    assert np.abs(samples).max(axis=0).min() > 0, "a channel is silent"
    norm = np.abs(samples).sum(axis=0).max()
    assert correction.gain_db == pytest.approx(20 * np.log10(norm), abs=1e-4)


def test_the_project_is_not_written(tmp_path: Path) -> None:
    """The engine works on a copy, so nothing appears next to the user's project."""
    project = _copy_fixture(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    calculate(project)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_peq_design_has_the_filters_of_each_channel(tmp_path: Path) -> None:
    """Minimum latency mode gives peaking filters, a gain and a delay per channel, and headroom."""
    correction = calculate_peq(_copy_fixture(tmp_path))

    assert correction.design_id
    assert correction.headroom_db == -5.0
    for channel in (correction.left, correction.right):
        assert len(channel.bands) == 32
        # within what a parametric EQ of the server accepts
        for band in channel.bands:
            assert 0 < band.frequency <= 100000
            assert 0.01 <= band.q <= 100
            assert -60 <= band.gain_db <= 60
    assert correction.left.gain_db == 0.0
    assert correction.right.gain_db == -0.13
    assert (correction.left.delay_ms, correction.right.delay_ms) == (0.0, 0.083)


def test_the_delay_keeps_the_difference_of_the_channels() -> None:
    """The delay filter delays R behind L by the design's difference, and is left out without one."""
    still = ChannelFilters(bands=(), gain_db=0.0, delay_ms=1.0)
    assert PeqCorrection(still, still, 0.0, None).delay() is None

    later = ChannelFilters(bands=(), gain_db=0.0, delay_ms=3.0)
    delay = PeqCorrection(still, later, 0.0, None).delay()
    assert delay is not None
    assert delay.sample_rate == 48000
    samples = _float_samples(delay.wav)
    # a 2 ms difference is 96 samples at 48 kHz
    assert np.diff(np.abs(samples).argmax(axis=0)) == 96
    # each channel passes the signal at unity gain
    assert samples.sum(axis=0) == pytest.approx(1.0, abs=1e-4)
    assert delay.gain_db == pytest.approx(20 * np.log10(np.abs(samples).sum(axis=0).max()), 1e-3)


@pytest.mark.parametrize("calculation", [calculate, calculate_peq])
def test_a_design_with_subwoofers_is_refused(tmp_path: Path, calculation: object) -> None:
    """A player has two channels: the 7.1.4 sample room with subwoofers cannot be applied."""
    project = tmp_path / "room.secs"
    engine = secs2r.Engine(tmp_path / "settings.json")
    engine.call(
        {
            "method": "project_new",
            "params": {"path": str(project), "preset": "2.0", "sub_count": 0, "sample": True},
        }
    )
    job = engine.call({"method": "sample_load", "params": {"positions": 2}})
    assert engine.wait_job(job["data"]["job"], 600)["status"] == "succeeded"
    engine.call({"method": "project_save", "params": {}})

    assert callable(calculation)
    with pytest.raises(CorrectionError, match=r"must be 2\.0"):
        calculation(project)


def test_a_missing_project_is_refused(tmp_path: Path) -> None:
    """A path that is not a project file says so."""
    with pytest.raises(CorrectionError, match=r"not a \.secs project"):
        calculate(tmp_path / "missing.secs")


def test_a_project_without_measurements_is_refused(tmp_path: Path) -> None:
    """The engine's refusal comes back as a CorrectionError."""
    project = _copy_fixture(tmp_path)
    shutil.rmtree(project.with_suffix(".irs"))
    with pytest.raises(CorrectionError):
        calculate(project)


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the stereo fixture project into the temporary folder."""
    shutil.copy(FIXTURES / "stereo.secs", tmp_path / "stereo.secs")
    shutil.copytree(FIXTURES / "stereo.irs", tmp_path / "stereo.irs")
    return tmp_path / "stereo.secs"


def _float_samples(data: bytes) -> np.ndarray:
    """Return the (frames, channels) samples of a WAV file, decoded by ffmpeg like the server."""
    probe = subprocess.run(
        # ffmpeg and ffprobe from PATH, as the server finds them
        ["ffprobe", "-v", "error", "-show_entries", "stream=channels,sample_rate,codec_name",  # noqa: S607
         "-of", "csv=p=0", "-"],
        input=data, capture_output=True, check=True,
    )  # fmt: skip
    assert probe.stdout.decode().strip() == "pcm_f32le,48000,2"
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le", "-"],  # noqa: S607
        input=data, capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, 2)
