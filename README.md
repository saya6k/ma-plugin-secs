# SECS 2R room correction for Music Assistant

A [Music Assistant](https://music-assistant.io) plugin that calculates the room correction of a
[SECS 2R](https://github.com/115dkk/Sound-Equalization-and-Correction-System-2R) project with the
SECS 2R engine ([`secs2r`](https://pypi.org/project/secs2r/)) and applies it to players in their
DSP settings, as a parametric EQ or as a convolution filter.

## How it works

1. Measure the room and set the project up in the SECS 2R application (its own measurement, REW,
   an mdat file or a Dirac Live project). Save the project: a `.secs` file with its `.irs` folder.
2. Put the project where the Music Assistant server can read it, for example in a shared folder
   of Home Assistant (`/share/room/stereo.secs`).
3. In Music Assistant, add the plugin, enter the full path of the project, choose the players and
   how to apply, and save. Then press **Calculate and apply**.

The plugin copies the project to a temporary folder and runs the SECS 2R calculation there. It
adds the result to the DSP settings of each player, after the filters already there, and switches
DSP on. Pressing the button again replaces the correction, also when the way to apply changed;
**Remove the correction** takes it out again.

| How to apply | What the player gets |
|---|---|
| **Parametric EQ** (default) | The design in SECS 2R's minimum latency mode, which uses no FIR: a parametric EQ with the peaking filters of each channel (32 per channel), the gain of each channel and the headroom. You see and can change the bands in the player's DSP settings. A delay between the channels, which the server's DSP has no filter for, is added as a small convolution filter. No latency, light on the CPU. |
| **Convolution** | The whole design in the project's mode (FIR, filters, gain, delay) as one stereo impulse response, in a convolution filter. More exact; heavier, and the low latency and standard modes add their latency. |

Impulse responses the plugin stores are named `SECS 2R <project> <design>`; those no player or
preset uses any more are removed, and others are never touched. A parametric EQ has no name, so
the plugin keeps the ones it added in `secs_room_correction.json` of the server's data folder.
One you changed after it was applied is yours: the plugin leaves it in place.

## Limits

- **2.0 only.** A player plays two channels, so the design must have only the paths L→L and R→R:
  no subwoofers, no MIMO. The plugin refuses other designs and names their paths.
- **Less exact as a parametric EQ.** Without FIR the correction is coarser: in the test room the
  share of the band within ±1 dB of the target rose from 14% to 39% for L as a parametric EQ,
  and to 55% with the low latency mode as a convolution. The server also runs its biquads in
  32-bit float, which misses the deepest narrow cuts in the bass by up to about 1 dB; in double
  precision the server's filters match the design within 0.05 dB.
- **48 kHz design.** SECS 2R designs at 48 kHz; the server resamples an impulse response to the
  stream's rate. A convolution keeps its shape and the balance of L and R, but at 44.1 kHz the
  overall level is about 0.4 dB higher, because the server normalises the resampled impulse
  response. The delay filter of the parametric EQ is a pulse limited to 18 kHz so that it keeps
  its level within 0.3 dB at every rate (a bare impulse would lose up to 8 dB); above 18 kHz it
  rolls off.
- **Saved settings.** The buttons use the saved project and players: save before pressing them.
- **Platforms.** `secs2r` 0.25.2 has wheels for Linux x86_64, macOS and Windows, but not for Linux
  arm64 (Raspberry Pi, Home Assistant Green and Yellow). The server installs requirements as
  wheels from PyPI only, so on arm64 it refuses to load the plugin, and the community catalog
  cannot list it until `secs2r` publishes an arm64 wheel. One built from the same source
  (manylinux 2_17 aarch64) passes all the tests of this repository in the arm64 image of the
  server, so adding the build to the release of `secs2r` is all that is missing.

## Development

Start from the [community provider template](https://github.com/music-assistant/community-provider-template)
workflow: `uv sync`, then `scripts/run.sh` starts a server that loads the plugin from this folder.

```sh
uv run pytest            # engine, provider and end-to-end tests (needs ffmpeg 8+)
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run codespell
```

`tests/fixtures/stereo.secs` is the L and R of the SECS 2R sample room;
`scripts/make_stereo_fixture.py` makes it again.

## Licence

AGPL-3.0-only, like the `secs2r` engine it runs. SECS 2R and SECS 2 are the work of their authors;
see the SECS 2R repository for its credits and licences.

---

## 한국어

SECS 2R 프로젝트(`.secs`와 옆의 `.irs` 폴더)의 룸 보정을 SECS 2R 엔진으로 계산해, Music Assistant
플레이어의 DSP 설정에 컨볼루션 필터로 넣는 플러그인입니다. 측정과 설정은 SECS 2R 앱에서 하고,
플러그인 설정에 프로젝트 경로와 플레이어를 저장한 뒤 **Calculate and apply**를 누릅니다.

- 적용 방식은 두 가지입니다. **PEQ**(기본)는 SECS 2R의 최소 지연 모드(FIR 없음)로 계산해 플레이어
  DSP의 PEQ로 넣고(채널마다 피킹 필터 32개, 채널 게인, 헤드룸), 채널 사이 딜레이만 작은 컨볼루션으로
  겁니다. MA 화면에서 밴드를 보고 고칠 수 있고 가볍습니다. **컨볼루션**은 프로젝트 모드의 설계 전체를
  IR 하나로 넣어 더 정확합니다.
- 플레이어는 2채널이므로 L→L, R→R만 있는 2.0 설계만 적용됩니다(서브·MIMO 불가).
- MA는 바이쿼드를 32비트 float로 계산해, 저역의 깊고 좁은 컷이 최대 1 dB쯤 덜 들어갑니다.
- 44.1 kHz 재생에서는 서버가 IR을 리샘플링하면서 컨볼루션의 전체 레벨이 약 0.4 dB 올라갑니다.
- `secs2r`에 Linux arm64 휠이 없어 라즈베리 파이 등에는 설치되지 않고, 커뮤니티 카탈로그 등록도
  그 휠이 PyPI에 올라온 뒤에 가능합니다. 같은 소스로 만든 arm64 휠로는 서버 arm64 이미지에서 이
  저장소의 테스트가 모두 통과합니다.
