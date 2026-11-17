"""Tests of the provider in this repository."""

from pathlib import Path

# the domain of the provider, which names its folder
DOMAIN = "secs_room_correction"
REPOSITORY = Path(__file__).parents[1]
FIXTURES = REPOSITORY / "tests" / "fixtures"
