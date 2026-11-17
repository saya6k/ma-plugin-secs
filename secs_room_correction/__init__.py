"""
SECS 2R room correction for Music Assistant.

Calculates the room correction of a SECS 2R project (made with the SECS 2R application, which
measures the room) with the SECS 2R engine, and applies it to players as a convolution filter in
their DSP settings. Music Assistant plays two channels, so the project must be a 2.0 design.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import (
    ConfigActionResult,
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.dsp import ConvolutionFilter, DSPConfig
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import CONF_PLAYER_DSP
from music_assistant.models.plugin import PluginProvider

from .engine import CorrectionError, StereoCorrection, calculate

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_PROJECT = "project"
CONF_PLAYERS = "players"
ACTION_APPLY = "apply"
ACTION_REMOVE = "remove"
# the impulse responses this provider stores carry this name, which is how it finds its own
IR_NAME_PREFIX = "SECS 2R"


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
        async with self._lock:
            try:
                correction = await asyncio.to_thread(calculate, Path(project))
            except CorrectionError as err:
                raise InvalidDataError(str(err)) from err
            ir = await self._store_ir(correction, Path(project).stem)
            for player_id in player_ids:
                config = self._without_own_filters(
                    self.mass.config.get_player_dsp_config(player_id)
                )
                config.filters.append(
                    ConvolutionFilter(enabled=True, ir_id=ir, gain=correction.gain_db)
                )
                config.enabled = True
                await self.mass.config.save_dsp_config(player_id, config)
            await self._remove_unused_irs()
        self.logger.info(
            "Applied the correction of %s (design %s) to %s",
            project,
            correction.design_id,
            ", ".join(player_ids),
        )
        return ConfigActionResult(
            message=f"Applied the room correction to {len(player_ids)} player(s)"
        )

    async def remove(self) -> ConfigActionResult:
        """Remove the correction of this provider from the chosen players."""
        player_ids = self._player_ids()
        async with self._lock:
            for player_id in player_ids:
                current = self.mass.config.get_player_dsp_config(player_id)
                config = self._without_own_filters(current)
                if len(config.filters) != len(current.filters):
                    await self.mass.config.save_dsp_config(player_id, config)
            await self._remove_unused_irs()
        return ConfigActionResult(
            message=f"Removed the room correction from {len(player_ids)} player(s)"
        )

    def _player_ids(self) -> list[str]:
        """Return the chosen players."""
        value = self.config.get_value(CONF_PLAYERS) or []
        return [str(player_id) for player_id in value] if isinstance(value, list) else []

    async def _store_ir(self, correction: StereoCorrection, project_name: str) -> str:
        """Store the impulse response with the server and return its id."""
        name = f"{IR_NAME_PREFIX} {project_name}"
        if correction.design_id:
            name += f" {correction.design_id[:8]}"
        record = await self.mass.config.upload_dsp_ir(
            name, base64.b64encode(correction.wav).decode("ascii")
        )
        return str(record["ir_id"])

    def _own_irs(self) -> set[str]:
        """Return the ids of the impulse responses this provider stored."""
        return {
            str(record["ir_id"])
            for record in self.mass.config.get_dsp_irs()
            if str(record.get("name", "")).startswith(IR_NAME_PREFIX)
        }

    def _without_own_filters(self, config: DSPConfig) -> DSPConfig:
        """Return the DSP settings without the convolution filters of this provider."""
        own = self._own_irs()
        return replace(
            config,
            filters=[
                dsp_filter
                for dsp_filter in config.filters
                if not (isinstance(dsp_filter, ConvolutionFilter) and dsp_filter.ir_id in own)
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
