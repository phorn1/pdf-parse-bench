## Project Overview

This is a synthetic PDF generator tool for creating benchmark datasets. The system generates HTML documents with randomized styling from JSON input data, then converts them to PDFs using Playwright and Chromium. The primary use case is generating diverse document layouts for PDF parsing benchmarks.


## Architecture

**Key Design Patterns:**
- Configuration constants are centralized in `config.py` to avoid magic numbers

The system expects `data.json` in the project root containing an array of objects with a "data" field containing the content to be rendered. It always alternates between type text and type formula.
Example content of the `data.json` file:
"""
[
    {
        "type": "text",
        "data": "Knowledge arrive position hope. Long political others city ground give. Air indeed all need least work. Between control service drop. Local already section. Service condition push today player nothing bag. Entire enough likely radio before tonight prevent. Structure decide tough as."
    },
    {
        "type": "formula",
        "data": "$$\\langle \\psi '_{r}|P_{3}|\\psi '_{r}\\rangle =0.$$"
    },
    {
        "type": "text",
        "data": "Win send edge second out physical. Religious great nearly real worry view agree fund. Last individual prepare large PM she central establish. Join success fight should north recognize shoulder. Stay government enough still late. Prepare agent else generation."
    },
    {
        "type": "formula",
        "data": "$$\\left({\\frac {\\partial g_{k}}{\\partial x_{i}}}\\right)\\forall _{k},_{i}$$"
    },
...
"""
