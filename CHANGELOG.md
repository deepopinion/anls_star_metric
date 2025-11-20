# Changelog

## Release vx.x.x ()
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

### Fixed

### Removed
- Outdated `tutorials` directory.
- `twine` from dev dependencies.
