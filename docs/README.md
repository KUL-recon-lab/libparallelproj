# Documentation

This directory contains the Sphinx documentation for parallelproj-backend.

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
2. Use Breathe to parse the XML
3. Build the Sphinx HTML documentation

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

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `api/` - API reference pages
- `requirements.txt` - Python dependencies
- `Makefile` - Build automation
