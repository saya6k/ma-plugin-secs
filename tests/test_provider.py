"""Tests of the provider, loaded by a real server."""

import asyncio
import base64
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from music_assistant_models.dsp import (
    ConvolutionFilter,
    DSPConfig,
    DSPFilter,
    GainFilter,
    ParametricEQFilter,
)
from music_assistant_models.enums import ProviderTier
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items.audio_format import AudioFormat

from music_assistant.constants import DSP_IRS_DIRNAME
from music_assistant.helpers.dsp import ComplexFilter, filter_to_ffmpeg_params
from music_assistant.mass import MusicAssistant
from secs_room_correction import (
    APPLIED_FILE,
    CONF_METHOD,
    CONF_PLAYERS,
    CONF_PROJECT,
    METHOD_CONVOLUTION,
    METHOD_PEQ,
    SecsRoomCorrection,
)
from secs_room_correction.engine import PEQ_MODE, calculate
from tests import DOMAIN, FIXTURES

PLAYER = "test_player"
RATE = 48000


async def test_provider_loads(mass: MusicAssistant) -> None:
    """The server loads the provider from this repository as a community provider."""
    assert mass.get_provider_manifest(DOMAIN).tier == ProviderTier.COMMUNITY


async def test_convolution_apply_replace_and_remove(mass: MusicAssistant, tmp_path: Path) -> None:
    """The convolution joins the player's own filters, replaces itself, and leaves on remove."""
    provider = await _provider(mass, tmp_path, method=METHOD_CONVOLUTION)
    own_filter = GainFilter(enabled=True, gain=-3.0)
    await mass.config.save_dsp_config(PLAYER, DSPConfig(enabled=False, filters=[own_filter]))

    await provider.apply()
    first = _filters(mass, ConvolutionFilter)
    assert len(first) == 1
    config = mass.config.get_player_dsp_config(PLAYER)
    assert config.enabled
    assert config.filters[0] == own_filter
    assert 0 < first[0].gain < 60

    # applying again replaces the filter, and the impulse response nothing uses is removed
    await provider.apply()
    second = _filters(mass, ConvolutionFilter)
    assert len(second) == 1
    assert second[0].ir_id != first[0].ir_id
    assert [ir["ir_id"] for ir in mass.config.get_dsp_irs()] == [second[0].ir_id]

    await provider.remove()
    assert mass.config.get_player_dsp_config(PLAYER).filters == [own_filter]
    assert mass.config.get_dsp_irs() == []


async def test_peq_apply_replace_and_remove(mass: MusicAssistant, tmp_path: Path) -> None:
    """The parametric EQ and its delay join the player's filters, replace themselves and leave."""
    provider = await _provider(mass, tmp_path, method=METHOD_PEQ)
    own_filter = GainFilter(enabled=True, gain=-3.0)
    await mass.config.save_dsp_config(PLAYER, DSPConfig(enabled=False, filters=[own_filter]))

    await provider.apply()
    config = mass.config.get_player_dsp_config(PLAYER)
    assert config.enabled
    own, peq, delay = config.filters
    assert own == own_filter
    assert isinstance(peq, ParametricEQFilter)
    assert peq.preamp == -5.0
    assert len(peq.bands) == 64
    assert {band.channel for band in peq.bands} == {"FL", "FR"}
    # R arrives 0.083 ms (4 samples) after L in the design
    assert isinstance(delay, ConvolutionFilter)

    await provider.apply()
    assert len(mass.config.get_player_dsp_config(PLAYER).filters) == 3
    assert len(mass.config.get_dsp_irs()) == 1

    await provider.remove()
    assert mass.config.get_player_dsp_config(PLAYER).filters == [own_filter]
    assert mass.config.get_dsp_irs() == []
    assert json.loads((Path(mass.storage_path) / APPLIED_FILE).read_text()) == {}


async def test_an_edited_peq_is_left_alone(mass: MusicAssistant, tmp_path: Path) -> None:
    """A parametric EQ the user changed after it was applied is theirs, and stays."""
    provider = await _provider(mass, tmp_path, method=METHOD_PEQ)
    await provider.apply()
    config = mass.config.get_player_dsp_config(PLAYER)
    peq = config.filters[0]
    assert isinstance(peq, ParametricEQFilter)
    edited = replace(peq, bands=[replace(peq.bands[0], gain=-1.0), *peq.bands[1:]])
    await mass.config.save_dsp_config(
        PLAYER, replace(config, filters=[edited, *config.filters[1:]])
    )

    await provider.remove()
    assert mass.config.get_player_dsp_config(PLAYER).filters == [edited]


async def test_switching_the_method_replaces_the_correction(
    mass: MusicAssistant, tmp_path: Path
) -> None:
    """Applying with the other method takes the first correction out."""
    provider = await _provider(mass, tmp_path, method=METHOD_PEQ)
    await provider.apply()
    provider = await _provider(mass, tmp_path, method=METHOD_CONVOLUTION)
    await provider.apply()
    (only,) = mass.config.get_player_dsp_config(PLAYER).filters
    assert isinstance(only, ConvolutionFilter)
    assert len(mass.config.get_dsp_irs()) == 1


async def test_other_impulse_responses_are_left_alone(mass: MusicAssistant, tmp_path: Path) -> None:
    """An impulse response the user uploaded is never removed by the provider."""
    provider = await _provider(mass, tmp_path, method=METHOD_CONVOLUTION)
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


async def test_server_plays_the_designed_convolution(mass: MusicAssistant, tmp_path: Path) -> None:
    """Through the server's own ffmpeg filters, the convolution sounds as SECS 2R designed it."""
    provider = await _provider(mass, tmp_path, method=METHOD_CONVOLUTION)
    await provider.apply()
    signal = (np.random.default_rng(0).standard_normal((RATE, 2)) * 0.05).astype("<f4")
    output = await asyncio.to_thread(_play, mass, signal)

    # SECS 2R renders each design as one impulse response through the same digital filters
    designed = await asyncio.to_thread(calculate, tmp_path / "stereo.secs")
    ir = _decode(designed.wav)
    expected = np.stack(
        [np.convolve(signal[:, c], ir[:, c])[: len(output)] for c in range(2)], axis=1
    )
    assert np.abs(output - expected).max() < 1e-5


@pytest.mark.parametrize(
    ("sample_format", "tolerance_db"),
    [
        # in double precision, the server's filters are the design: the values carried over
        ("aformat=sample_fmts=dblp", 0.05),
        # the server processes in 32-bit float, where ffmpeg's biquads miss the deepest cuts in
        # the bass: by 1.0 dB on arm64 and by 3.1 dB on x86_64 with the same ffmpeg 9.0.1
        ("", 3.5),
    ],
)
async def test_server_plays_the_designed_peq(
    mass: MusicAssistant, tmp_path: Path, sample_format: str, tolerance_db: float
) -> None:
    """Through the server's own ffmpeg filters, the parametric EQ sounds as SECS 2R designed it."""
    provider = await _provider(mass, tmp_path, method=METHOD_PEQ)
    await provider.apply()
    impulse = np.zeros((65536, 2), dtype="<f4")
    impulse[0] = 1.0
    output = await asyncio.to_thread(_play, mass, impulse, sample_format)

    designed = _decode((await asyncio.to_thread(calculate, tmp_path / "stereo.secs", PEQ_MODE)).wav)
    # the delay keeps R as far behind L as the design has it
    assert np.diff(np.abs(output).argmax(axis=0)) == np.diff(np.abs(designed).argmax(axis=0))
    size = 1 << 17
    frequencies = np.fft.rfftfreq(size, 1 / RATE)
    # the delay pulse passes up to 18 kHz
    band = (frequencies > 20) & (frequencies < 16000)
    played = np.abs(np.fft.rfft(output, size, axis=0))[band]
    expected = np.abs(np.fft.rfft(designed, size, axis=0))[band]
    assert np.abs(20 * np.log10(played / expected)).max() < tolerance_db


async def _provider(
    mass: MusicAssistant,
    tmp_path: Path,
    *,
    project: str | None = None,
    players: list[str] | None = None,
    method: str = METHOD_PEQ,
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
    provider.config.values[CONF_METHOD].value = method
    return cast("SecsRoomCorrection", provider)


def _filters[T: DSPFilter](mass: MusicAssistant, kind: type[T]) -> list[T]:
    """Return the filters of one kind of the test player."""
    return [f for f in mass.config.get_player_dsp_config(PLAYER).filters if isinstance(f, kind)]


def _play(mass: MusicAssistant, signal: np.ndarray, sample_format: str = "") -> np.ndarray:
    """
    Return the stereo signal after the ffmpeg filters the server makes of the player's DSP.

    :param sample_format: An ffmpeg filter that converts the samples before the DSP.
    """
    ir_dir = str(Path(mass.storage_path) / DSP_IRS_DIRNAME)
    audio_format = AudioFormat(sample_rate=RATE)
    graph: list[str] = [f"[0:a]{sample_format or 'anull'}[in]"]
    inputs: list[str] = []
    current = "[in]"
    for dsp_filter in mass.config.get_player_dsp_config(PLAYER).filters:
        for part in filter_to_ffmpeg_params(dsp_filter, audio_format, ir_dir=ir_dir):
            label = f"[s{len(graph)}]"
            if isinstance(part, ComplexFilter):
                (ir_input,) = part.inputs
                inputs += ["-i", ir_input.path]
                ir = f"[ir{len(inputs)}]"
                graph.append(f"[{len(inputs) // 2}:a]{ir_input.filters}{ir}")
                graph.append(f"{current}{ir}{part.body}{label}")
            else:
                graph.append(f"{current}{part}{label}")
            current = label
    played = subprocess.run(  # noqa: S603 - the inputs are the server's own impulse responses
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", "-",  # noqa: S607
         *inputs, "-filter_complex", ";".join(graph), "-map", current, "-f", "f32le", "-"],
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
