#!/usr/bin/env bash
# Render docs/PROJECT_REPORT.md to a submission PDF.
#
#   bash scripts/build_report_pdf.sh [output.pdf]
#
# Requires pandoc and a TeX distribution providing xelatex (MacTeX on macOS).
# XeLaTeX rather than pdflatex because the report contains Unicode the latter
# cannot typeset (em dashes, x-signs, and the box-drawing diagram).
#
# Two transformations are applied to a temporary copy; the committed markdown
# is never modified:
#
#   1. Box-drawing characters are folded to ASCII. Even monospace fonts that
#      nominally contain these glyphs render them at inconsistent widths under
#      XeTeX, which breaks the alignment the diagram depends on. ASCII +-|v
#      always lines up.
#   2. Tables are typeset at \footnotesize. Pandoc derives LaTeX column widths
#      from the markdown source, so wide tables (the five-column environment
#      comparison, the topic interface) overflow and print columns on top of
#      one another at full size.
#
# Contributor: Carlos Gonzales
set -euo pipefail

SRC="docs/PROJECT_REPORT.md"
OUT="${1:-docs/CMPE249_Gonzales_Project_Report.pdf}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
command -v pandoc >/dev/null || { echo "pandoc not installed" >&2; exit 1; }
command -v xelatex >/dev/null || { echo "xelatex not installed (MacTeX)" >&2; exit 1; }

python3 - "$SRC" "$TMP/report.md" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
fold = {
    '┌': '+', '┐': '+', '└': '+', '┘': '+',
    '┬': '+', '┴': '+', '├': '+', '┤': '+',
    '┼': '+', '─': '-', '│': '|',
    '▶': '>', '▼': 'v', '→': '->', '←': '<-',
}
text = open(src, encoding='utf-8').read()
for k, v in fold.items():
    text = text.replace(k, v)
open(dst, 'w', encoding='utf-8').write(text)
PY

cat > "$TMP/header.tex" <<'TEX'
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
\AtBeginEnvironment{tabular}{\footnotesize}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize Carlos Gonzales, CMPE 249}
\fancyhead[R]{\footnotesize FCOS3D in ROS 2}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
TEX

pandoc "$TMP/report.md" -o "$OUT" \
    --pdf-engine=xelatex \
    --toc --toc-depth=2 \
    -V geometry:margin=1in \
    -V fontsize=10pt \
    -V monofont="Menlo" \
    -V monofontoptions="Scale=0.80" \
    -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
    -H "$TMP/header.tex"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
