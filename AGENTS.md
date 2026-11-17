# AGENTS.md

This repository is a community provider for Music Assistant, an async Python server that plays
music from streaming services and libraries on the user's own speakers. Servers install it from the
community catalog; the README links the rules of the catalog and says how to develop, test and
release.

## Never write code that

- gets around an account, a subscription tier, a region, a service's limit of streams, or copy
  protection;
- downloads, exports or archives music, or writes audio of a service to disk;
- hands a service's own audio address to anything outside the server (a media item, an API result
  or a route);
- calls a service's API without a throttler, asks again for what the cache holds, keeps sending
  while the service asks to wait, or walks its catalogue for one user action;
- sends the user's data anywhere but the service it connects to, loads code from somewhere else
  while it runs, or does something the manifest's description does not mention;
- writes into the provider's own folder.

## How to work here

- `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`, `uv run mypy` and
  `uv run codespell` must pass.
- Never block the event loop; wrap blocking calls in `asyncio.to_thread`.
- Requirements in `manifest.json` are package names with version specifiers only, and come as
  wheels for the Python version and platform of the server: a package that only comes as source
  cannot be installed.
- Follow the server's code style: Sphinx docstrings with `:param:`, the summary of a multi-line
  docstring on the line after the quotes, comments only where the code is not obvious.
- The server pin is the `music-assistant` line in `pyproject.toml`; change it only when asked.
