"""Render the small, readable overview panels used by HexaBee Hub cards.

The published PNGs are checked into the repository so staging a release does not
need Pillow or a particular font installation. Run this script only when the
verified release statistics or card design change.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "src/pct4b/assets/hub-overviews"
FONT_PATH = "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    try:
        result = ImageFont.truetype(FONT_PATH, size)
        result.set_variation_by_name(weight)
        return result
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


# Values below are grounded in the archived release_stats.json files and the
# accompanying manuscript. Training-record counts are not unique-source counts.
PANELS = {
    "model": (
        "FINAL TRANSLATOR",
        "HexaBee-4B",
        [
            ("6", "LANGUAGES", "FA / EN / AR / ZH / HE / ES"),
            ("30", "DIRECTIONS", "All non-identity routes"),
            ("62.03", "BLEU SCORE", "Equal-direction macro"),
        ],
    ),
    "multilingual-base": (
        "MERGED CHECKPOINT · PHASE 02",
        "Multilingual Base",
        [
            ("02", "SUPERVISION TASKS", "Answering + answer-text MT"),
            ("6", "LANGUAGES", "Evaluation language set"),
            ("4B", "GEMMA 3 BASE", "Intermediate merge"),
        ],
    ),
    "lora-pt": (
        "ADAPTER · PHASE 01",
        "Wikipedia PT LoRA",
        [
            ("01", "TRAINING PHASE", "Continued pretraining"),
            ("256", "LORA RANK", "Archived adapter config"),
            ("4B", "GEMMA 3 BASE", "Exact checkpoint lineage"),
        ],
    ),
    "lora-sft": (
        "ADAPTER · PHASE 02",
        "Multilingual SFT LoRA",
        [
            ("02", "TRAINING TASKS", "Answering + answer-text MT"),
            ("30.15M", "SFT RECORDS", "Archived phase-2 corpus"),
            ("256", "LORA RANK", "Archived adapter config"),
        ],
    ),
    "lora-translation": (
        "ADAPTER · PHASE 03",
        "Translation LoRA",
        [
            ("03", "TRAINING PHASE", "Final specialization"),
            ("658,274", "SFT RECORDS", "Forward + reverse"),
            ("30", "DIRECTIONS", "Six-language coverage"),
        ],
    ),
    "data-pt": (
        "DATASET · PHASE 01",
        "Wikipedia Foundations",
        [
            ("11.4M", "RAW TEXT ROWS", "Not unique articles"),
            ("5", "CONFIGURATIONS", "fa · en · ar · zh · arz"),
            ("01", "TRAINING PHASE", "Continued pretraining"),
        ],
    ),
    "data-sft": (
        "DATASET · PHASE 02",
        "Multilingual Answering",
        [
            ("19.46M", "ANSWERING ROWS", "Question to answer"),
            ("10.70M", "TRANSLATION ROWS", "Answer text across languages"),
            ("30.15M", "TOTAL RECORDS", "Expanded SFT corpus"),
        ],
    ),
    "data-translation": (
        "DATASET · PHASE 03",
        "Bidirectional Translation",
        [
            ("329,146", "TRAIN PAIRS", "Aligned content items"),
            ("36,572", "TEST PAIRS", "Original held-out split"),
            ("658,274", "SFT RECORDS", "Forward + reverse"),
        ],
    ),
    "prompts": (
        "DATASET · PROMPT INVENTORY",
        "The Prompt Hive",
        [
            ("341,142", "IMPORTED ROWS", "Historical sampling mass"),
            ("6", "COMPONENTS", "Language · style · translation"),
            ("6", "LANGUAGES", "Multilingual instructions"),
        ],
    ),
    "outputs": (
        "DATASET · PREDICTION ARCHIVE",
        "Translation Archive",
        [
            ("110,616", "STORED ROWS", "Full + judged subsets"),
            ("3", "SYSTEMS", "HexaBee + 2 baselines"),
            ("30", "DIRECTIONS", "Matched source identities"),
        ],
    ),
    "evaluation": (
        "DATASET · EVALUATION",
        "Evaluation Atlas",
        [
            ("30", "DIRECTIONS", "Six-language matrix"),
            ("3", "SYSTEMS", "Paired comparisons"),
            ("10,000", "BOOTSTRAPS", "Direction-stratified"),
        ],
    ),
}

COLORS = [
    ("#FFF0CD", "#D8891F"),
    ("#D8F5F0", "#128A83"),
    ("#E8E5FF", "#6558C7"),
]
NAVY = "#17263D"
MUTED = "#4F6075"


def fitted_font(draw: ImageDraw.ImageDraw, value: str, max_width: int) -> ImageFont.FreeTypeFont:
    size = 84
    while size > 48:
        candidate = font(size, "Bold")
        if draw.textbbox((0, 0), value, font=candidate)[2] <= max_width:
            return candidate
        size -= 2
    return font(size, "Bold")


def draw_hexagon(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int, fill: str) -> None:
    import math

    points = [
        (cx + radius * math.cos(math.pi / 3 * i), cy + radius * math.sin(math.pi / 3 * i))
        for i in range(6)
    ]
    draw.polygon(points, fill=fill)


def render(key: str, eyebrow: str, title: str, facts: list[tuple[str, str, str]]) -> None:
    image = Image.new("RGB", (1600, 520), "#F7F8FB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 12), fill="#C3944C")
    draw.text((60, 43), eyebrow, font=font(26, "Bold"), fill="#9B6A22")
    draw.text((60, 84), title, font=font(52, "Bold"), fill=NAVY)
    draw.line((60, 155, 1540, 155), fill="#DDE3EA", width=2)

    for i, (value, label, detail) in enumerate(facts):
        x = 60 + i * 500
        background, accent = COLORS[i]
        draw.rounded_rectangle((x + 4, 181, x + 484, 464), radius=30, fill="#E7EBF1")
        draw.rounded_rectangle((x, 175, x + 480, 456), radius=30, fill=background)
        draw.rounded_rectangle((x, 175, x + 480, 188), radius=8, fill=accent)
        draw_hexagon(draw, x + 422, 235, 25, accent)
        draw_hexagon(draw, x + 422, 235, 13, background)
        draw.text((x + 34, 211), value, font=fitted_font(draw, value, 365), fill=NAVY)
        draw.text((x + 34, 330), label, font=font(30, "Bold"), fill=accent)
        draw.text((x + 34, 385), detail, font=font(23), fill=MUTED)

    draw.text(
        (60, 480),
        "HEXABEE  /  ARCHIVED RESEARCH ARTIFACTS  /  SEE CARD FOR PROVENANCE AND LIMITATIONS",
        font=font(18, "Medium"),
        fill="#657286",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / f"{key}.png", optimize=True)


if __name__ == "__main__":
    for panel_key, (panel_eyebrow, panel_title, panel_facts) in PANELS.items():
        render(panel_key, panel_eyebrow, panel_title, panel_facts)
