"""
Run the SECS 2R engine on a project and turn its result into filters for a stereo player.

Two ways: one stereo impulse response of the whole design (convolution), or the parametric EQ,
gain and delay of each channel of a design in SECS 2R's minimum latency mode, which uses no FIR.
Everything here blocks; the provider calls it in a thread.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import secs2r

# how long the calculation of one project may take
JOB_TIMEOUT = 600.0
# WAVE_FORMAT_IEEE_FLOAT
_WAV_FLOAT = 3
_WAV_EXTENSIBLE = 0xFFFE
# SECS 2R's mode without FIR: peaking filters, a gain and a delay per channel
PEQ_MODE = "zero"
# the sample rate SECS 2R designs at
DESIGN_RATE = 48000
_NOT_STEREO = (
    "A Music Assistant player plays two channels, so the design must be 2.0: "
    "only L->L and R->R, without subwoofers or MIMO. This design has {}."
)
# the delay pulse: a Kaiser-windowed sinc that passes up to 18 kHz. A bare impulse changes shape
# when the server resamples it to the stream's rate, so the normalization of afir would change
# its level by up to 8 dB; a pulse limited below the Nyquist frequency of 44.1 kHz keeps its
# shape and its level within 0.3 dB at every rate
_PULSE_CUTOFF = 19000.0
_PULSE_HALF = 48
_PULSE_BETA = 9.0
# room before the first pulse, so its ringing after resampling is not cut off
_PULSE_LEAD = 64
# afir does not finish with a very short impulse response
_MIN_IR_FRAMES = 256
_EQAPO_PEAK = re.compile(
    r"Filter:\s+ON\s+PK\s+Fc\s+(\S+)\s+Hz\s+Gain\s+(\S+)\s+dB\s+Q\s+(\S+)", re.IGNORECASE
)
_EQAPO_PREAMP = re.compile(r"Preamp:\s+(\S+)\s+dB", re.IGNORECASE)
_EQAPO_DELAY = re.compile(r"Delay:\s+(\S+)\s+ms", re.IGNORECASE)
_EQAPO_CHANNELS = {"VML": "L", "VMR": "R", "ALL": None}


class CorrectionError(Exception):
    """The project cannot be turned into a correction for a stereo player."""


@dataclass(frozen=True)
class StereoCorrection:
    """A stereo impulse response with the gain that undoes the server's normalization."""

    wav: bytes
    sample_rate: int
    # ffmpeg's afir (irnorm=1, irlink) divides the IR by the largest L1 norm of its channels;
    # this gain in dB multiplies it back, so the correction keeps its designed level
    gain_db: float
    design_id: str | None


@dataclass(frozen=True)
class PeqBand:
    """A peaking filter (RBJ biquad), as SECS 2R and Music Assistant both define it."""

    frequency: float
    q: float
    gain_db: float


@dataclass(frozen=True)
class ChannelFilters:
    """The filters of one channel, applied in this order: peaking filters, gain, delay."""

    bands: tuple[PeqBand, ...]
    gain_db: float
    delay_ms: float


@dataclass(frozen=True)
class PeqCorrection:
    """The filters of a design in minimum latency mode."""

    left: ChannelFilters
    right: ChannelFilters
    # the attenuation SECS 2R puts before the filters, for the largest boost
    headroom_db: float
    design_id: str | None

    def delay(self) -> StereoCorrection | None:
        """
        Return a stereo impulse response that delays the channels, or None without delay.

        The channels keep their delay relative to each other; a delay both share is left out.
        """
        delays = [channel.delay_ms * DESIGN_RATE / 1000 for channel in (self.left, self.right)]
        base = min(delays)
        shifts = [delay - base for delay in delays]
        if max(shifts) < 1:
            return None
        frames = max(_MIN_IR_FRAMES, int(_PULSE_LEAD + max(shifts) + _PULSE_HALF + 1))
        ir = np.stack([_pulse(_PULSE_LEAD + shift, frames) for shift in shifts], axis=1)
        return StereoCorrection(
            wav=_write_float_wav(ir.astype(np.float32), DESIGN_RATE),
            sample_rate=DESIGN_RATE,
            gain_db=20 * float(np.log10(np.abs(ir).sum(axis=0).max())),
            design_id=self.design_id,
        )


def calculate(project: Path, mode: str | None = None) -> StereoCorrection:
    """
    Calculate the correction of a SECS 2R project and return it as a stereo impulse response.

    The project is copied first, so the engine never writes next to the user's file.

    :param project: The `.secs` file; its measurements are in the `.irs` folder next to it.
    :param mode: A SECS 2R correction mode to use instead of the project's.
    """
    with _export(project, "convolution", mode) as export:
        return _stereo_correction(export)


def calculate_peq(project: Path) -> PeqCorrection:
    """
    Calculate the correction of a SECS 2R project in minimum latency mode, which uses no FIR.

    :param project: The `.secs` file; its measurements are in the `.irs` folder next to it.
    """
    with _export(project, "equalizer_apo", PEQ_MODE) as export:
        return _peq_correction(export)


@contextmanager
def _export(project: Path, kind: str, mode: str | None) -> Iterator[Path]:
    """Calculate a copy of the project and yield the folder of one of its exports."""
    if project.suffix != ".secs" or not project.is_file():
        raise CorrectionError(f"{project} is not a .secs project file")
    with tempfile.TemporaryDirectory(prefix="secs_room_correction_") as tmp:
        work = Path(tmp)
        copy = _copy_project(project, work / "project")
        engine = secs2r.Engine(work / "settings.json")
        export = work / "export"
        export.mkdir()
        try:
            engine.call({"method": "project_open", "params": {"path": str(copy)}})
            if mode is not None:
                settings = engine.call({"method": "calibration_settings_get"})["data"]
                settings["mode"] = mode
                engine.call(
                    {"method": "calibration_settings_set", "params": {"settings": settings}}
                )
            _run(engine, {"method": "calibrate", "params": {}})
            _run(engine, {"method": "export", "params": {"kind": kind, "folder": str(export)}})
        except secs2r.SecsError as err:
            raise CorrectionError(f"SECS 2R: {err.message}") from err
        yield export


def _copy_project(project: Path, target: Path) -> Path:
    """Copy the project file and its `.irs` folder, and return the copied project file."""
    target.mkdir()
    shutil.copy2(project, target / project.name)
    irs = project.with_suffix(".irs")
    if irs.is_dir():
        shutil.copytree(irs, target / irs.name)
    return target / project.name


def _run(engine: secs2r.Engine, request: dict[str, Any]) -> None:
    """Start a job and wait until it succeeds."""
    job = engine.call(request)["data"]["job"]
    finished = engine.wait_job(job, JOB_TIMEOUT)
    if finished.get("status") != "succeeded":
        error = finished.get("error") or {}
        detail = error.get("detail") or error.get("code") or finished.get("status")
        raise CorrectionError(f"SECS 2R {request['method']} failed: {detail}")


def _stereo_correction(export: Path) -> StereoCorrection:
    """Combine the L and R paths of a convolution export into one stereo impulse response."""
    routing = json.loads((export / "routing.json").read_text(encoding="utf-8"))
    paths = {(path["input"], path["output"]): path["wav"] for path in routing["paths"]}
    if set(paths) != {("L", "L"), ("R", "R")}:
        raise CorrectionError(_NOT_STEREO.format(", ".join(f"{i}->{o}" for i, o in sorted(paths))))
    rate = int(routing["sample_rate"])
    left = _read_mono_float(export / paths["L", "L"], rate)
    right = _read_mono_float(export / paths["R", "R"], rate)
    length = max(len(left), len(right))
    ir = np.zeros((length, 2), dtype=np.float32)
    ir[: len(left), 0] = left
    ir[: len(right), 1] = right
    norm = float(np.abs(ir).sum(axis=0).max())
    if norm <= 0:
        raise CorrectionError("The correction is silent")
    return StereoCorrection(
        wav=_write_float_wav(ir, rate),
        sample_rate=rate,
        gain_db=20 * float(np.log10(norm)),
        design_id=_design(export).get("design_id"),
    )


def _peq_correction(export: Path) -> PeqCorrection:
    """Read the filters of each channel from an Equalizer APO export of a 2.0 design."""
    design = _design(export)
    inputs, outputs = design.get("inputs"), design.get("outputs")
    if inputs != ["L", "R"] or outputs != ["L", "R"]:
        raise CorrectionError(_NOT_STEREO.format(f"inputs {inputs} and outputs {outputs}"))
    config = (export / "config.txt").read_text(encoding="utf-8-sig")
    includes = re.findall(r"^Include:\s*(.+?)\s*$", config, re.MULTILINE)
    if len(includes) != 1:
        raise CorrectionError("The Equalizer APO export does not include one filter file")
    headroom = sum(float(value) for value in _EQAPO_PREAMP.findall(config))
    filters = (export / includes[0].replace("\\", "/")).read_text(encoding="utf-8-sig")
    left, right = _channel_filters(filters)
    return PeqCorrection(
        left=left, right=right, headroom_db=headroom, design_id=design.get("design_id")
    )


def _channel_filters(text: str) -> tuple[ChannelFilters, ChannelFilters]:
    """Return the filters of L and R from an Equalizer APO filter file of SECS 2R."""
    bands: dict[str, list[PeqBand]] = {"L": [], "R": []}
    gains = {"L": 0.0, "R": 0.0}
    delays = {"L": 0.0, "R": 0.0}
    channel: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("channel:"):
            name = line.split(":", 1)[1].strip().upper()
            if name not in _EQAPO_CHANNELS:
                raise CorrectionError(f"The design filters channel {name}, which is not L or R")
            channel = _EQAPO_CHANNELS[name]
        elif channel is None:
            raise CorrectionError(f"The design has a filter for all channels: {line}")
        elif peak := _EQAPO_PEAK.fullmatch(line):
            frequency, gain, q = (float(value) for value in peak.groups())
            bands[channel].append(PeqBand(frequency=frequency, q=q, gain_db=gain))
        elif preamp := _EQAPO_PREAMP.fullmatch(line):
            gains[channel] += float(preamp.group(1))
        elif delay := _EQAPO_DELAY.fullmatch(line):
            delays[channel] += float(delay.group(1))
        else:
            raise CorrectionError(f"The design has a filter a parametric EQ cannot hold: {line}")
    left, right = (
        ChannelFilters(bands=tuple(bands[c]), gain_db=gains[c], delay_ms=delays[c]) for c in "LR"
    )
    return left, right


def _design(export: Path) -> dict[str, Any]:
    """Return the design record of an export, empty when there is none."""
    design = export / "secs-design.json"
    if not design.is_file():
        return {}
    data = json.loads(design.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _pulse(center: float, frames: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return a band-limited pulse at a (fractional) sample position, with a DC gain of one."""
    offset = np.arange(frames) - center
    pulse = 2 * _PULSE_CUTOFF / DESIGN_RATE * np.sinc(2 * _PULSE_CUTOFF / DESIGN_RATE * offset)
    inside = np.abs(offset) <= _PULSE_HALF
    window = np.zeros(frames)
    window[inside] = np.i0(_PULSE_BETA * np.sqrt(1 - (offset[inside] / _PULSE_HALF) ** 2))
    pulse *= window / np.i0(_PULSE_BETA)
    normalized: np.ndarray[Any, np.dtype[np.float64]] = pulse / pulse.sum()
    return normalized


def _read_mono_float(path: Path, rate: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return the samples of a mono 32-bit float WAV file at the given sample rate."""
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CorrectionError(f"{path.name} is not a WAV file")
    fmt: tuple[int, ...] | None = None
    tag = 0
    offset = 12
    while offset + 8 <= len(data):
        chunk, size = data[offset : offset + 4], struct.unpack_from("<I", data, offset + 4)[0]
        body = offset + 8
        if chunk == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", data, body)
            tag = fmt[0]
            if tag == _WAV_EXTENSIBLE and size >= 26:
                # the format of an extensible header is the start of its sub-format GUID
                tag = struct.unpack_from("<H", data, body + 24)[0]
        elif chunk == b"data":
            if fmt is None:
                break
            _, channels, file_rate, _, _, bits = fmt
            if channels != 1 or file_rate != rate or bits != 32 or tag != _WAV_FLOAT:
                raise CorrectionError(f"{path.name} is not mono 32-bit float at {rate} Hz")
            return np.frombuffer(data, dtype="<f4", count=size // 4, offset=body).astype(np.float32)
        offset = body + size + (size & 1)
    raise CorrectionError(f"{path.name} has no audio")


def _write_float_wav(samples: np.ndarray[Any, np.dtype[np.float32]], rate: int) -> bytes:
    """Return a 32-bit float WAV file of the given (frames, channels) samples."""
    frames, channels = samples.shape
    payload = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", _WAV_FLOAT, channels, rate, rate * channels * 4, channels * 4, 32)
    # a float WAV carries a fact chunk with the number of frames
    fact = struct.pack("<I", frames)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"fact" + struct.pack("<I", len(fact)) + fact
        + b"data" + struct.pack("<I", len(payload)) + payload
    )  # fmt: skip
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
