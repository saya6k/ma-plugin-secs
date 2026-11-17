"""
Run the SECS 2R engine on a project and turn its result into one stereo impulse response.

Everything here blocks; the provider calls it in a thread.
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
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


def calculate(project: Path) -> StereoCorrection:
    """
    Calculate the correction of a SECS 2R project and return it as a stereo impulse response.

    The project is copied first, so the engine never writes next to the user's file.

    :param project: The `.secs` file; its measurements are in the `.irs` folder next to it.
    """
    if project.suffix != ".secs" or not project.is_file():
        raise CorrectionError(f"{project} is not a .secs project file")
    with tempfile.TemporaryDirectory(prefix="secs_room_correction_") as tmp:
        work = Path(tmp)
        copy = _copy_project(project, work / "project")
        engine = secs2r.Engine(work / "settings.json")
        try:
            engine.call({"method": "project_open", "params": {"path": str(copy)}})
            _run(engine, {"method": "calibrate", "params": {}})
            export = work / "export"
            export.mkdir()
            _run(
                engine,
                {"method": "export", "params": {"kind": "convolution", "folder": str(export)}},
            )
        except secs2r.SecsError as err:
            raise CorrectionError(f"SECS 2R: {err.message}") from err
        return _stereo_correction(export)


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
        found = ", ".join(f"{i}->{o}" for i, o in sorted(paths))
        raise CorrectionError(
            "A Music Assistant player plays two channels, so the design must be 2.0: "
            f"only L->L and R->R, without subwoofers or MIMO. This design has {found}."
        )
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
    design_id = None
    design = export / "secs-design.json"
    if design.is_file():
        design_id = json.loads(design.read_text(encoding="utf-8")).get("design_id")
    return StereoCorrection(
        wav=_write_float_wav(ir, rate),
        sample_rate=rate,
        gain_db=20 * float(np.log10(norm)),
        design_id=design_id,
    )


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
