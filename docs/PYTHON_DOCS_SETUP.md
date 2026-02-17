## Building Documentation for Python Nanobind Interface

I've set up complete documentation for both C and Python APIs. Here's what was done:

### 1. ✅ Added Docstrings to Python Bindings

Added comprehensive NumPy-style docstrings to all 5 functions in `python/bindings.cpp`:
- `joseph3d_fwd` / `joseph3d_back`
- `joseph3d_tof_sino_fwd` / `joseph3d_tof_sino_back`
- `joseph3d_tof_lm_fwd`

### 2. ✅ Updated Sphinx Configuration

Modified `docs/conf.py` to:
- Add Python build directory to sys.path for module import
- Enable autodoc to extract Python docstrings

### 3. ✅ Created Documentation Structure

- `docs/python_api.rst` - Python API documentation using autodoc
- `docs/c_api.rst` - C API documentation using Breathe
- `docs/index.rst` - Main page linking both APIs

### 4. ✅ Updated Build Process

Modified `docs/Makefile` to:
- Build the Python module (`make parallelproj_backend`)
- Generate Doxygen XML
- Build Sphinx HTML

---

## 🚀 Quick Start

```bash
# Install Python dependencies
pip install -r docs/requirements.txt

# Build everything (C + Python docs)
cd docs
make html

# View result
open _build/html/index.html
```

The documentation will show:
- **Python API** - With all parameter types, descriptions, and notes from C++ docstrings
- **C API** - With Doxygen-formatted documentation from header files

Both use the clean Read the Docs theme with full navigation and search.

---

## 📝 Notes

- Docstrings are embedded in the nanobind bindings using raw string literals (`R"pbdoc(...)pbdoc"`)
- Sphinx's autodoc extracts these at build time from the compiled Python module
- Changes to docstrings require rebuilding the Python module (`make html` does this automatically)
- The Python module must be successfully built for autodoc to work
