# Contributing to napari-simpleannotate

Thank you for your interest in contributing to napari-simpleannotate!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/napari-simpleannotate.git
cd napari-simpleannotate
```

2. Install in development mode:
```bash
pip install -e .[testing]
```

## Testing

Run tests with pytest:
```bash
pytest -v --color=yes --cov=napari_simpleannotate --cov-report=xml
```

Or use tox for testing across multiple environments:
```bash
tox
```

## Code Quality

This project uses ruff and black for code formatting and linting.

Run linting:
```bash
# Check with ruff
ruff check src/

# Auto-fix with ruff
ruff check --fix src/

# Format with black (line length: 120)
black src/
```

## Making Changes

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure tests pass

3. Commit your changes with clear commit messages

4. Push to your fork and create a Pull Request

## Releasing a New Version

**Important**: PyPI releases are automated via GitHub Actions and triggered by version tags.

### Release Process

1. Update the version number in `setup.py` or `pyproject.toml`

2. Commit the version change:
```bash
git add setup.py
git commit -m "Bump version to v0.2.0"
```

3. Push to main branch:
```bash
git push origin main
```

4. Create and push a version tag:
```bash
git tag v0.2.0
git push origin v0.2.0
```

### What Happens Automatically

When you push a tag with the format `v*` (e.g., `v0.2.0`, `v1.0.0`):

1. **GitHub Actions** (`.github/workflows/release.yml`) is triggered
2. **Tests** run on all supported platforms (Linux, macOS, Windows) and Python versions (3.8, 3.9, 3.10)
3. **Package** is built and validated
4. **GitHub Release** is created with auto-generated changelog
5. **PyPI publication** happens automatically if all tests pass

### Important Notes

- **Regular pushes to `main` or `develop` do NOT trigger PyPI releases**
- Only version tags (`v*` format) trigger the release workflow
- The release will fail if tests don't pass
- Make sure the version in your code matches the tag version

## CI/CD Workflows

This project has two GitHub Actions workflows:

### ci.yml
- **Triggers**: Push/PR to `main` or `develop` branches
- **Actions**: Run tests, linting, and build checks
- **Does NOT publish to PyPI**

### release.yml
- **Triggers**: Push of tags matching `v*` pattern
- **Actions**: Run tests, create GitHub release, publish to PyPI
- **Requires**: All tests must pass before publishing

## Questions?

If you have questions about contributing, please open an issue on GitHub.
