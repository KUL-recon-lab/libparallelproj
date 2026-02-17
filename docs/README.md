# Documentation

This directory contains the Sphinx documentation for parallelproj-backend, including both C and Python APIs.

## Building the Documentation

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

You also need Doxygen installed:

```bash
# macOS
brew install doxygen

# Ubuntu/Debian
sudo apt-get install doxygen

# Fedora
sudo dnf install doxygen
```

### Build HTML Documentation

Simply run:

```bash
make html
```

This will:
1. Generate Doxygen XML from the C header files
2. Build the Python extension module (parallelproj_backend)
3. Use Breathe to parse the Doxygen XML
4. Use autodoc to extract Python docstrings
5. Build the Sphinx HTML documentation

The documentation will be available at `_build/html/index.html`.

### View the Documentation

Open in your browser:

```bash
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Clean Build

To clean the generated documentation:

```bash
make clean
```

## Documentation Structure

The documentation includes:

- **Python API** - Extracted from nanobind docstrings in `python/bindings.cpp`
- **C API** - Extracted from Doxygen comments in `include/parallelproj.h`

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `python_api.rst` - Python API reference (uses autodoc)
- `c_api.rst` - C API reference (uses Breathe)
- `requirements.txt` - Python dependencies
- `Makefile` - Build automation
