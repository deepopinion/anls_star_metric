# Changelog

## Release vx.x.x ()
### Added

### Changed

### Fixed

### Removed


## Release v1.0.0 (2026-01-07)
### Added
- `ruff` for sorting of imports
- `py.typed` marker file for PEP 561 compliance
- Project-related metadata to `pyproject.toml`.
- `mypy` type checking in CI

### Changed
- Enabled dependency grouping in `dependabot` to collect all dependency updates in one PR rather than one per dependency.

### Fixed
- Return type inconsistency in `_levenshtein_distance`

### Removed
- Support for Python `3.9` dropped, since it is end of life.


## Release v0.1.0 (2025-11-21)
### Added
- Add release pipeline
- Linting and Formatting via `ruff` (also in CI)
- Tested compatibility with Python `3.9` to `3.14`
- `dependabot` for dependency updates

### Changed
- Switch from `hatchling` to `uv_build`
- Switch from `twine` to `uv publish`
- Introduce `src` layout for the package
- Switch from `pip` to `uv`.

### Removed
- Outdated `tutorials` directory.
- `twine` from dev dependencies.
- `benchmark` directory (moved to separate repo since it uses private Otera packages)
