# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile doxygen

# Generate Doxygen XML first
doxygen:
	@echo "Generating Doxygen XML..."
	@cd .. && mkdir -p build && cd build && \
		cmake .. -DBUILD_DOCS=ON -DBUILD_PYTHON=ON -DBUILD_TESTS=OFF -DUSE_CUDA=OFF && \
		make doxygen-docs && \
		make parallelproj_backend

# Build HTML with Doxygen and Python module first
html: doxygen
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
	@echo "Open $(BUILDDIR)/html/index.html in a browser."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
