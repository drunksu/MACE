# Contributing to MACE-RL

We welcome contributions from the community! This document provides guidelines for contributing to the MACE-RL project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/your-org/mace-rl/issues) page
- Include a clear description of the issue
- Provide steps to reproduce if applicable
- Include system information (OS, Python version, etc.)

### Feature Requests

- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Consider whether it aligns with the project's goals

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run the test suite** to ensure nothing is broken
   ```bash
   pytest tests/
   ```
7. **Commit your changes** with descriptive commit messages
   ```bash
   git commit -m "Add feature: description of changes"
   ```
8. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Open a Pull Request** against the main repository

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation for Development

```bash
# Clone your fork
git clone https://github.com/your-username/mace-rl.git
cd mace-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_module.py

# Run with coverage report
pytest --cov=mace_rl tests/
```

### Code Quality Tools

```bash
# Format code
black mace_rl/ tests/ examples/

# Sort imports
isort mace_rl/ tests/ examples/

# Lint code
flake8 mace_rl/ tests/ examples/

# Type checking
mypy mace_rl/ --ignore-missing-imports
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)
- Use f-strings for string formatting (Python 3.6+)

### Documentation

- Document all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings where helpful
- Update README.md for user-facing changes

### Testing

- Write unit tests for new functionality
- Aim for high test coverage
- Use descriptive test names
- Mock external dependencies

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature")
- Limit first line to 72 characters
- Reference issues in the body (e.g., "Fixes #123")

## Project Structure

Understand the project layout before making changes:

```
mace_rl/
├── data/           # Dataset loading and preprocessing
├── features/       # Microstructure feature extraction
├── environment/    # Execution environment
├── models/        # Neural network architectures
├── training/      # Training loops and algorithms
├── utils/         # Utility functions
└── verl/          # VerL integration
```

## Areas for Contribution

### High Priority

- Performance optimizations
- Additional microstructure features
- New RL algorithms
- Enhanced visualization tools
- More extensive test coverage

### Medium Priority

- Additional datasets
- Docker support
- CI/CD improvements
- Documentation improvements
- Example notebooks

### Low Priority

- New normalization methods
- Alternative flow architectures
- Additional baseline methods

## Review Process

1. **Automated Checks**: CI runs tests, linting, and type checking
2. **Maintainer Review**: At least one maintainer reviews the PR
3. **Feedback**: Address any requested changes
4. **Merge**: Once approved, a maintainer merges the PR

## License

By contributing to MACE-RL, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

Open an issue or join the [GitHub Discussions](https://github.com/your-org/mace-rl/discussions).

Thank you for contributing to MACE-RL! 🚀
