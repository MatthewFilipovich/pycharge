# Contributing

Thank you for your interest in contributing to **PyCharge**!  
We welcome contributions from the community to help improve the project.

## Reporting bugs, asking questions, or suggesting features

If you encounter a bug, have a question, or would like to suggest a new feature, please [open an issue on GitHub](https://github.com/MatthewFilipovich/pycharge/issues) with a clear description.  
Including screenshots, minimal code examples, or error logs is appreciated.

## Development setup

To prepare your environment for development and testing:

1. **Fork the repository on [GitHub](https://github.com/MatthewFilipovich/pycharge/fork)**.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/pycharge.git
   cd pycharge
   ```
3. **Install PyCharge in development mode**:
   ```bash
   pip install -e '.[dev]'
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

Pre-commit hooks will automatically run the following tools:

- **ruff** — for code formatting and linting
- **pyright** — for static type checking

## Submitting a pull request

Submit your changes by opening a pull request from your fork.

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes and commit them**.
3. **Run the tests**:
   ```bash
   pytest
   ```
4. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a pull request** against the `main` branch.  
   Include a clear description and reference any related issues if applicable.

## Editing the documentation

If you want to update the documentation, you can preview it locally before submitting a pull request.

1. **Install documentation dependencies**:

   ```bash
   pip install -r docs/requirements-docs.txt
   ```

2. **Serve the documentation locally**:

   ```bash
   sphinx-autobuild docs/source docs/build/html
   ```

The site will be available at [http://localhost:8000/](http://localhost:8000/).
