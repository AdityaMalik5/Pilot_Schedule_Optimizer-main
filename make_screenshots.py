"""
Generate 11 code screenshots for project report using Pygments + Pillow.
"""
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
from pygments.styles import get_style_by_name
import os

OUT = os.path.join(os.path.dirname(__file__), "report_screenshots")
os.makedirs(OUT, exist_ok=True)

SECTIONS = [
    ("01_imports_airport_config", "app.py", 1, 88),
    ("02_pso_optimizer", "app.py", 624, 708),
    ("03_ga_optimizer", "app.py", 713, 820),
    ("04_schedule_optimizer_slots", "app.py", 875, 998),
    ("05_normalize_columns", "test4.py", 47, 111),
    ("06_clean_flight_data", "test4.py", 429, 600),
    ("07_schedule_tuner_class", "test4.py", 605, 717),
    ("08_cascading_delay_analyzer", "test4.py", 722, 803),
    ("09_enhanced_optimizer", "test4.py", 865, 1000),
    ("10_pilot_fatigue_predictor", "generate_ready_figures.py", 45, 100),
    ("11_greedy_swap_optimizer", "generate_ready_figures.py", 213, 273),
]

BASE = os.path.dirname(__file__)

def read_lines(filename, start, end):
    path = os.path.join(BASE, filename)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    snippet = "".join(lines[start-1:end])
    return snippet

formatter = ImageFormatter(
    style=get_style_by_name("monokai"),
    font_name="Courier New",
    font_size=14,
    line_numbers=True,
    line_number_start=1,
    image_pad=20,
    line_pad=4,
)

for name, fname, start, end in SECTIONS:
    code = read_lines(fname, start, end)
    # Reset line_number_start per snippet
    fmt = ImageFormatter(
        style=get_style_by_name("monokai"),
        font_name="Courier New",
        font_size=14,
        line_numbers=True,
        line_number_start=start,
        image_pad=20,
        line_pad=4,
    )
    result = highlight(code, PythonLexer(), fmt)
    out_path = os.path.join(OUT, f"{name}.png")
    with open(out_path, "wb") as f:
        f.write(result)
    print(f"✓ {name}.png")

print(f"\nAll screenshots saved to: {OUT}/")
