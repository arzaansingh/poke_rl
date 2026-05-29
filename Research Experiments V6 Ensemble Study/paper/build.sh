#!/usr/bin/env bash
# Build the V6 paper PDF.
#
# Usage:
#   ./build.sh         — clean build + check
#   ./build.sh clean   — remove build artifacts only
#   ./build.sh figures — regenerate figures from eval JSONs, then build
#
# Prereqs:
#   - TeX Live or MacTeX installed (pdflatex, latexmk, bibtex)
#   - neurips_2024.sty in this directory (auto-downloaded if missing)

set -euo pipefail

cd "$(dirname "$0")"

ensure_neurips_sty() {
  if [[ -f neurips_2024.sty ]]; then return; fi
  echo "Downloading neurips_2024.sty..."
  curl -fsSL -o neurips_2024.sty \
    "https://raw.githubusercontent.com/official-Auralin/Multimodal-World-Simulation-Architecture/main/neurips_2024.sty"
}

regen_figures() {
  echo "Regenerating figures..."
  (cd .. && python analyze_paper_figures.py)
}

clean() {
  echo "Cleaning build artifacts..."
  rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc
}

build() {
  ensure_neurips_sty
  echo "Building main.pdf..."
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

  echo ""
  echo "=== Build summary ==="
  ls -la main.pdf
  echo ""
  echo "Warnings:"
  grep -E "Warning|undefined" main.log | grep -v "^Package: " | head -10 || echo "  (none)"
  echo ""
  echo "Open main.pdf to review."
}

case "${1:-build}" in
  clean)   clean ;;
  figures) regen_figures && build ;;
  build|"") build ;;
  *)
    echo "Usage: $0 [build|clean|figures]"
    exit 1
    ;;
esac
