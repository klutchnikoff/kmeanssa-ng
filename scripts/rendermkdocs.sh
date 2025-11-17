#!/bin/bash

# Render README.qmd to README.md
quarto render docs-src/README.qmd --to gfm -o README.md --execute

# Generate performance documentation from benchmark results
python scripts/generate_benchmark_docs.py

# Render all Quarto documents as a project
# This respects _quarto.yml settings (freeze, seed, etc.)
cd docs-src
quarto render --to gfm
cd ..

# Copy API documentation
cp -r docs-src/api docs/

# Remove GFM duplicate files to avoid mkdocs warnings
find docs/api -name '*-gfm.md' -delete