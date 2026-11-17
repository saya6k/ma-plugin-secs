"""
SECS 2R room correction for Music Assistant.

Calculates the room correction of a SECS 2R project (made with the SECS 2R application, which
measures the room) with the SECS 2R engine, and applies it to players in their DSP settings: as a
parametric EQ (SECS 2R's minimum latency mode, which uses no FIR) or as a convolution filter of
the whole design. Music Assistant plays two channels, so the project must be a 2.0 design.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from music_assistant_models.config_entries import (
    ConfigActionResult,
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.dsp import (
    AudioChannel,
    ConvolutionFilter,
    DSPConfig,
    DSPFilter,
    ParametricEQBand,
    ParametricEQBandType,
    ParametricEQFilter,
)
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import CONF_PLAYER_DSP
from music_assistant.models.plugin import PluginProvider

from .engine import (
    CorrectionError,
    PeqCorrection,
    StereoCorrection,
    calculate,
    calculate_peq,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_PROJECT = "project"
CONF_PLAYERS = "players"
CONF_METHOD = "method"
METHOD_PEQ = "peq"
METHOD_CONVOLUTION = "convolution"
ACTION_APPLY = "apply"
ACTION_REMOVE = "remove"
# the impulse responses this provider stores carry this name, which is how it finds its own
IR_NAME_PREFIX = "SECS 2R"
# a parametric EQ has no name, so the provider keeps the ones it added, by player, in this file
# of the server's data folder, and finds them again by their content
APPLIED_FILE = "secs_room_correction.json"


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize the provider for one settings entry."""
    return SecsRoomCorrection(mass, manifest, config, set())


class SecsRoomCorrection(PluginProvider):
    """Applies the room correction of a SECS 2R project to players."""

    # one calculation or change of the DSP settings at a time
    _lock = asyncio.Lock()

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the project, the players and the buttons that apply or remove the correction."""
        players = [
            ConfigValueOption(player.name, player.player_id)
            for player in self.mass.players.iter_players(return_unavailable=True)
        ]
        return (
            # not required: the server adds the plugin before anyone can type a path, and the
            # button that needs it says when it is missing
            ConfigEntry(
                key=CONF_PROJECT, type=ConfigEntryType.STRING, default_value="", required=False
            ),
            ConfigEntry(
                key=CONF_PLAYERS,
                type=ConfigEntryType.STRING,
                multi_value=True,
                options=players,
                default_value=[],
                required=False,
            ),
            ConfigEntry(
                key=CONF_METHOD,
                type=ConfigEntryType.STRING,
                options=[
                    ConfigValueOption("Parametric EQ (minimum latency mode)", METHOD_PEQ),
                    ConfigValueOption("Convolution (the project's mode)", METHOD_CONVOLUTION),
                ],
                default_value=METHOD_PEQ,
                required=False,
            ),
            ConfigEntry(
                key=ACTION_APPLY, type=ConfigEntryType.ACTION, action=ACTION_APPLY, required=False
            ),
            ConfigEntry(
                key=ACTION_REMOVE,
                type=ConfigEntryType.ACTION,
                action=ACTION_REMOVE,
                required=False,
            ),
        )

    async def handle_config_action(
        self, action: str
    ) -> tuple[ConfigEntry, ...] | ConfigActionResult | None:
        """Calculate and apply the correction, or remove it from the players."""
        if action == ACTION_APPLY:
            return await self.apply()
        if action == ACTION_REMOVE:
            return await self.remove()
        return await super().handle_config_action(action)

    async def apply(self) -> ConfigActionResult:
        """Calculate the correction of the project and apply it to the chosen players."""
        project = str(self.config.get_value(CONF_PROJECT) or "").strip()
        player_ids = self._player_ids()
        if not project:
            raise InvalidDataError("Set the path of the SECS 2R project first")
        if not player_ids:
            raise InvalidDataError("Choose at least one player first")
        method = self.config.get_value(CONF_METHOD) or METHOD_PEQ
        name = Path(project).stem
        async with self._lock:
            try:
                if method == METHOD_CONVOLUTION:
                    correction = await asyncio.to_thread(calculate, Path(project))
                    filters: list[DSPFilter] = [await self._convolution(correction, name)]
                    design_id = correction.design_id
                else:
                    peq = await asyncio.to_thread(calculate_peq, Path(project))
                    filters = [_parametric_eq(peq)]
                    if delay := peq.delay():
                        filters.append(await self._convolution(delay, f"{name} delay"))
                    design_id = peq.design_id
            except CorrectionError as err:
                raise InvalidDataError(str(err)) from err
            applied = await self._read_applied()
            for player_id in player_ids:
                current = self.mass.config.get_player_dsp_config(player_id)
                config = self._without_own_filters(current, applied.get(player_id, []))
                config.filters.extend(filters)
                config.enabled = True
                await self.mass.config.save_dsp_config(player_id, config)
                applied[player_id] = [f.to_dict() for f in filters if not _is_convolution(f)]
            await self._write_applied(applied)
            await self._remove_unused_irs()
        self.logger.info(
            "Applied the %s correction of %s (design %s) to %s",
            method,
            project,
            design_id,
            ", ".join(player_ids),
        )
        return ConfigActionResult(
            message=f"Applied the room correction to {len(player_ids)} player(s)"
        )

    async def remove(self) -> ConfigActionResult:
        """Remove the correction of this provider from the chosen players."""
        player_ids = self._player_ids()
        async with self._lock:
            applied = await self._read_applied()
            for player_id in player_ids:
                current = self.mass.config.get_player_dsp_config(player_id)
                config = self._without_own_filters(current, applied.pop(player_id, []))
                if len(config.filters) != len(current.filters):
                    await self.mass.config.save_dsp_config(player_id, config)
            await self._write_applied(applied)
            await self._remove_unused_irs()
        return ConfigActionResult(
            message=f"Removed the room correction from {len(player_ids)} player(s)"
        )

    def _player_ids(self) -> list[str]:
        """Return the chosen players."""
        value = self.config.get_value(CONF_PLAYERS) or []
        return [str(player_id) for player_id in value] if isinstance(value, list) else []

    async def _convolution(self, correction: StereoCorrection, name: str) -> ConvolutionFilter:
        """Store the impulse response with the server and return a filter that plays it."""
        name = f"{IR_NAME_PREFIX} {name}"
        if correction.design_id:
            name += f" {correction.design_id[:8]}"
        record = await self.mass.config.upload_dsp_ir(
            name, base64.b64encode(correction.wav).decode("ascii")
        )
        return ConvolutionFilter(enabled=True, ir_id=str(record["ir_id"]), gain=correction.gain_db)

    async def _read_applied(self) -> dict[str, list[dict[str, Any]]]:
        """Return the parametric EQs this provider added, by player."""
        path = Path(self.mass.storage_path) / APPLIED_FILE

        def read() -> dict[str, list[dict[str, Any]]]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
            return data if isinstance(data, dict) else {}

        return await asyncio.to_thread(read)

    async def _write_applied(self, applied: dict[str, list[dict[str, Any]]]) -> None:
        """Keep the parametric EQs this provider added, by player."""
        path = Path(self.mass.storage_path) / APPLIED_FILE
        text = json.dumps({player: filters for player, filters in applied.items() if filters})
        await asyncio.to_thread(path.write_text, text, encoding="utf-8")

    def _own_irs(self) -> set[str]:
        """Return the ids of the impulse responses this provider stored."""
        return {
            str(record["ir_id"])
            for record in self.mass.config.get_dsp_irs()
            if str(record.get("name", "")).startswith(IR_NAME_PREFIX)
        }

    def _without_own_filters(self, config: DSPConfig, applied: list[dict[str, Any]]) -> DSPConfig:
        """
        Return the DSP settings without the filters this provider added.

        Those are its convolution filters, and the parametric EQs it added as they were added:
        one the user changed since is theirs now, and stays.
        """
        own = self._own_irs()
        return replace(
            config,
            filters=[
                dsp_filter
                for dsp_filter in config.filters
                if not (isinstance(dsp_filter, ConvolutionFilter) and dsp_filter.ir_id in own)
                and dsp_filter.to_dict() not in applied
            ],
        )

    async def _remove_unused_irs(self) -> None:
        """Remove the impulse responses of this provider that no player or preset uses."""
        used: set[str] = set()
        # the stored settings of every player, also of one that is not loaded right now
        stored: dict[str, object] = self.mass.config.get(CONF_PLAYER_DSP, {})
        configs = [self.mass.config.get_player_dsp_config(player_id) for player_id in stored]
        configs += [preset.config for preset in await self.mass.config.get_dsp_presets()]
        for config in configs:
            used.update(
                dsp_filter.ir_id
                for dsp_filter in config.filters
                if isinstance(dsp_filter, ConvolutionFilter)
            )
        for ir_id in self._own_irs() - used:
            await self.mass.config.remove_dsp_ir(ir_id)


def _parametric_eq(correction: PeqCorrection) -> ParametricEQFilter:
    """Return the parametric EQ of a design: its bands, channel gains and headroom."""
    bands = [
        ParametricEQBand(
            frequency=band.frequency,
            q=band.q,
            gain=band.gain_db,
            type=ParametricEQBandType.PEAK,
            enabled=True,
            channel=channel,
        )
        for channel, filters in (
            (AudioChannel.FL, correction.left),
            (AudioChannel.FR, correction.right),
        )
        for band in filters.bands
    ]
    return ParametricEQFilter(
        enabled=True,
        preamp=correction.headroom_db,
        per_channel_preamp={
            AudioChannel.FL: correction.left.gain_db,
            AudioChannel.FR: correction.right.gain_db,
        },
        bands=bands,
    )


def _is_convolution(dsp_filter: DSPFilter) -> bool:
    """Return whether a filter is a convolution, which the provider finds by its impulse response."""
    return isinstance(dsp_filter, ConvolutionFilter)
