#!/bin/bash

# Render README.qmd to README.md
quarto render docs-src/README.qmd --to gfm -o README.md --execute

# Clean up old markdown files in docs/
find docs/ -maxdepth 1 -type f -name "*.md" -delete

# Render all Quarto documents as a project
# This respects _quarto.yml settings (freeze, seed, etc.)
cd docs-src
quarto render --to gfm --no-cache
cd ..

# Copy API documentation
cp -r docs-src/api docs/

# Remove GFM duplicate files to avoid mkdocs warnings
find docs/api -name '*-gfm.md' -delete