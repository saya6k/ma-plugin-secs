"""Tests of the provider, loaded by a real server."""

import asyncio
import base64
import shutil
import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from music_assistant_models.dsp import ConvolutionFilter, DSPConfig, GainFilter
from music_assistant_models.enums import ProviderTier
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items.audio_format import AudioFormat

from music_assistant.constants import DSP_IRS_DIRNAME
from music_assistant.helpers.dsp import ComplexFilter, filter_to_ffmpeg_params
from music_assistant.mass import MusicAssistant
from secs_room_correction import CONF_PLAYERS, CONF_PROJECT, SecsRoomCorrection
from secs_room_correction.engine import calculate
from tests import DOMAIN, FIXTURES

PLAYER = "test_player"


async def test_provider_loads(mass: MusicAssistant) -> None:
    """The server loads the provider from this repository as a community provider."""
    assert mass.get_provider_manifest(DOMAIN).tier == ProviderTier.COMMUNITY


async def test_apply_replace_and_remove(mass: MusicAssistant, tmp_path: Path) -> None:
    """The correction joins the player's own filters, replaces itself, and leaves on remove."""
    provider = await _provider(mass, tmp_path)
    own_filter = GainFilter(enabled=True, gain=-3.0)
    await mass.config.save_dsp_config(PLAYER, DSPConfig(enabled=False, filters=[own_filter]))

    await provider.apply()
    first = _convolutions(mass)
    assert len(first) == 1
    config = mass.config.get_player_dsp_config(PLAYER)
    assert config.enabled
    assert config.filters[0] == own_filter
    assert 0 < first[0].gain < 60

    # applying again replaces the filter, and the impulse response nothing uses is removed
    await provider.apply()
    second = _convolutions(mass)
    assert len(second) == 1
    assert second[0].ir_id != first[0].ir_id
    assert [ir["ir_id"] for ir in mass.config.get_dsp_irs()] == [second[0].ir_id]

    await provider.remove()
    assert _convolutions(mass) == []
    assert mass.config.get_player_dsp_config(PLAYER).filters == [own_filter]
    assert mass.config.get_dsp_irs() == []


async def test_other_impulse_responses_are_left_alone(mass: MusicAssistant, tmp_path: Path) -> None:
    """An impulse response the user uploaded is never removed by the provider."""
    provider = await _provider(mass, tmp_path)
    data = (FIXTURES / "stereo.irs" / "L_P1.wav").read_bytes()
    record = await mass.config.upload_dsp_ir("my room", base64.b64encode(data).decode())
    await provider.apply()
    await provider.remove()
    assert [ir["ir_id"] for ir in mass.config.get_dsp_irs()] == [record["ir_id"]]


async def test_without_project_or_players(mass: MusicAssistant, tmp_path: Path) -> None:
    """The buttons say what is missing instead of failing later."""
    provider = await _provider(mass, tmp_path, players=[])
    with pytest.raises(InvalidDataError, match="player"):
        await provider.apply()
    provider = await _provider(mass, tmp_path, project="")
    with pytest.raises(InvalidDataError, match="project"):
        await provider.apply()


async def _provider(
    mass: MusicAssistant,
    tmp_path: Path,
    *,
    project: str | None = None,
    players: list[str] | None = None,
) -> SecsRoomCorrection:
    """Add the provider with the stereo fixture project and the test player."""
    if project is None:
        shutil.copy(FIXTURES / "stereo.secs", tmp_path / "stereo.secs")
        shutil.copytree(FIXTURES / "stereo.irs", tmp_path / "stereo.irs", dirs_exist_ok=True)
        project = str(tmp_path / "stereo.secs")
    if not mass.get_provider(DOMAIN):
        await mass.config.setup_provider(DOMAIN)
    provider = mass.get_provider(DOMAIN)
    # the server imports the provider as music_assistant.providers.<domain>, a module of its
    # own, so its class is not the one imported here
    assert provider is not None
    assert type(provider).__name__ == SecsRoomCorrection.__name__
    provider.config.values[CONF_PROJECT].value = project
    provider.config.values[CONF_PLAYERS].value = [PLAYER] if players is None else players
    return cast("SecsRoomCorrection", provider)


def _convolutions(mass: MusicAssistant) -> list[ConvolutionFilter]:
    """Return the convolution filters of the test player."""
    return [
        f
        for f in mass.config.get_player_dsp_config(PLAYER).filters
        if isinstance(f, ConvolutionFilter)
    ]


async def test_server_plays_the_designed_correction(mass: MusicAssistant, tmp_path: Path) -> None:
    """Through the server's own ffmpeg filters, the correction sounds as SECS 2R designed it."""
    provider = await _provider(mass, tmp_path)
    await provider.apply()
    (convolution,) = _convolutions(mass)
    ir_dir = str(Path(mass.storage_path) / DSP_IRS_DIRNAME)
    chain = filter_to_ffmpeg_params(convolution, AudioFormat(sample_rate=48000), ir_dir=ir_dir)
    complex_filter, volume = chain
    assert isinstance(complex_filter, ComplexFilter)
    assert isinstance(volume, str)
    (ir_input,) = complex_filter.inputs

    rng = np.random.default_rng(0)
    signal = (rng.standard_normal((48000, 2)) * 0.05).astype("<f4")
    output = await asyncio.to_thread(
        _play,
        signal,
        ir_input.path,
        f"[1:a]{ir_input.filters}[ir];[0:a][ir]{complex_filter.body},{volume}",
    )

    designed = await asyncio.to_thread(calculate, tmp_path / "stereo.secs")
    ir = _decode(designed.wav)
    expected = np.stack(
        [np.convolve(signal[:, c], ir[:, c])[: len(output)] for c in range(2)], axis=1
    )
    assert np.abs(output - expected).max() < 1e-5


def _play(signal: np.ndarray, ir_path: str, graph: str) -> np.ndarray:
    """Return the stereo signal after the ffmpeg filter graph that takes the IR as second input."""
    played = subprocess.run(  # noqa: S603 - the IR path is the server's own
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", "-",  # noqa: S607
         "-i", ir_path, "-filter_complex", graph, "-f", "f32le", "-"],
        input=signal.tobytes(), capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(played.stdout, dtype="<f4").reshape(-1, 2).astype(np.float64)


def _decode(data: bytes) -> np.ndarray:
    """Return the (frames, 2) samples of a stereo WAV file, decoded by ffmpeg."""
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le", "-"],  # noqa: S607
        input=data, capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, 2).astype(np.float64)
