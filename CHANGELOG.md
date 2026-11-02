# Changelog

## Release vx.x.x ()
### Added

### Changed
- Pinned `munkres` to `>=1.1.4,<2`, the last releases from its original maintainer. The PyPI project changed hands in October 2026, and 2.0.0/3.0.0 come from the new maintainer. On ties, per-key scores (`return_key_scores=True`) follow 1.1.4's tie-breaking again; total scores are unchanged.
- **BREAKING**: Invalid input *types* now raise `TypeError` instead of `ValueError`. `TypeError` is not a subclass of `ValueError`, so `except ValueError` no longer catches these. Affects unsupported leaf types and mismatched container types. `ValueError` is still raised for invalid *values* (e.g. empty ground truth tuple, tuple used in a prediction).

### Fixed
- Various linting errors after updating to `ruff v0.16.0`.

### Removed


## Release v1.0.1 (2026-01-12)
### Fixed
- `py.typed` needs to be included in the release as well.


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
