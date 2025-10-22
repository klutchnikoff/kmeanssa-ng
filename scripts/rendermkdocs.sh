#!/bin/bash

mkdir -p docs
rm -f docs/*.md
for qmd_file in docs-src/*.qmd; do
  quarto render "$qmd_file" --to gfm --output-dir ../docs
done
cp -r docs-src/api docs/
