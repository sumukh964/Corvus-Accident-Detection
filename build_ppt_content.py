# CORVUS presentation -- stage 1: content build (theme, text, images, charts).
# Animations/transitions are injected in a second pass (add_animations.py)
# so the content layer can be verified independently first.
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.oxml.ns import qn

ROOT = r"C:\Users\shafa\Projects\CORVUS"
IMG_ACCIDENT = r"C:\Users\shafa\OneDrive\Pictures\Screenshots\Accident Detection.png"
IMG_NORMAL   = r"C:\Users\shafa\OneDrive\Pictures\Screenshots\Normal Detectiom.png"
IMG_DISCORD  = r"C:\Users\shafa\Downloads\Discord Alert.PNG"
IMG_TELEGRAM = r"C:\Users\shafa\Downloads\Telegram Alert.PNG"
IMG_ARCH     = ROOT + r"\arch_diagram.png"

BG      = RGBColor(0x0A, 0x0A, 0x0D)
PANEL   = RGBColor(0x12, 0x18, 0x1F)
RED     = RGBColor(0xE4, 0x23, 0x2B)
RED_DIM = RGBColor(0x5A, 0x14, 0x18)
AMBER   = RGBColor(0xFF, 0x8C, 0x1A)
GREEN   = RGBColor(0x3A, 0xDB, 0x76)
WHITE   = RGBColor(0xF2, 0xF2, 0xF2)
GREY    = RGBColor(0x9A, 0xA2, 0xAC)
DGREY   = RGBColor(0x55, 0x5D, 0x66)

FONT_HEAD = "Arial"
FONT_BODY = "Segoe UI"
FONT_MONO = "Consolas"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]

_slide_counter = [0]

def new_slide():
    s = prs.slides.add_slide(BLANK)
    _slide_counter[0] += 1
    # background fill
    bg = s.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    return s

def add_rect(slide, x, y, w, h, color, line=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if line:
        sh.line.color.rgb = color
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh

def add_text(slide, x, y, w, h, text, size=18, color=WHITE, bold=False, italic=False,
             font=FONT_BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
             wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
    return tb

def add_bullets(slide, x, y, w, h, items, size=16, color=WHITE, font=FONT_BODY,
                 bullet_color=RED, line_spacing=1.15, space_after=10, bold_first=False):
    """items: list of str, or (str, size) tuples for sub-levels via leading tabs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        sub = item.startswith("\t")
        clean = item.lstrip("\t")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        marker = "›  " if not sub else "–  "
        r1 = p.add_run()
        r1.text = ("     " if sub else "") + marker
        r1.font.size = Pt(size - (2 if sub else 0))
        r1.font.color.rgb = bullet_color
        r1.font.name = font
        r1.font.bold = True
        r2 = p.add_run()
        r2.text = clean
        r2.font.size = Pt(size - (2 if sub else 0))
        r2.font.color.rgb = color if not sub else GREY
        r2.font.name = font
    return tb

def header(slide, kicker, title, number=None):
    """CORVUS-style header: small red kicker label, bold title, thin red rule."""
    if number is not None:
        add_text(slide, Inches(12.35), Inches(0.28), Inches(0.8), Inches(0.4),
                  f"{number:02d}", size=14, color=DGREY, font=FONT_MONO, align=PP_ALIGN.RIGHT)
    add_text(slide, Inches(0.55), Inches(0.32), Inches(8), Inches(0.35),
              kicker.upper(), size=13, color=RED, bold=True, font=FONT_MONO)
    add_text(slide, Inches(0.5), Inches(0.62), Inches(11.5), Inches(0.9),
              title, size=30, color=WHITE, bold=True, font=FONT_HEAD)
    add_rect(slide, Inches(0.55), Inches(1.42), Inches(12.25), Pt(2.2), RED)

def footer(slide, text="CORVUS  |  Dept. of AI & ML, MUSE, UoM"):
    add_text(slide, Inches(0.55), Inches(7.12), Inches(8), Inches(0.3),
              text, size=9, color=DGREY, font=FONT_MONO)

def kpi_block(slide, x, y, w, value, label, value_color=RED, size=36):
    add_text(slide, x, y, w, Inches(0.75), value, size=size, color=value_color, bold=True,
              font=FONT_MONO, align=PP_ALIGN.CENTER, wrap=False)
    add_text(slide, x, y + Inches(0.85), w, Inches(0.4), label.upper(), size=11,
              color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)

def picture_framed(slide, path, x, y, w, h, border=RED):
    pic = slide.shapes.add_picture(path, x, y, height=h)
    # center-crop-ish: constrain width if overflow, else keep aspect via height-only sizing
    if pic.width > w:
        pic.width = w
        pic.height = h
    frame = add_rect(slide, x - Pt(1.5), y - Pt(1.5), pic.width + Pt(3), pic.height + Pt(3), border)
    frame.fill.background()
    frame.line.color.rgb = border
    frame.line.width = Pt(1.5)
    slide.shapes._spTree.remove(frame._element)
    slide.shapes._spTree.insert(list(slide.shapes._spTree).index(pic._element), frame._element)
    return pic

# ============================================================ SLIDE 1 -- Title/cover
s = new_slide()
add_rect(s, 0, 0, SLIDE_W, Inches(0.12), RED)
add_rect(s, 0, Inches(7.38), SLIDE_W, Inches(0.12), RED)
add_text(s, Inches(0.8), Inches(0.45), Inches(11.7), Inches(1.0),
          "UNIVERSITY OF MYSORE", size=15, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.8), Inches(0.78), Inches(11.7), Inches(0.5),
          "Mysore University School of Engineering, Manasagangotri, Mysuru", size=12,
          color=DGREY, font=FONT_BODY, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.8), Inches(1.08), Inches(11.7), Inches(0.4),
          "AICTE Approved  |  NAAC 'A' Grade", size=11, color=DGREY, font=FONT_MONO,
          align=PP_ALIGN.CENTER)

add_text(s, Inches(1.0), Inches(2.05), Inches(11.3), Inches(1.0),
          "CORVUS", size=64, color=RED, bold=True, font=FONT_HEAD, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.0), Inches(2.95), Inches(11.3), Inches(0.5),
          "COLLISION ONSET RECOGNITION  ·  VEHICULAR  ·  UNIFIED  ·  SURVEILLANCE",
          size=14, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_rect(s, Inches(4.5), Inches(3.55), Inches(4.3), Pt(1.6), RED)
add_text(s, Inches(1.3), Inches(3.75), Inches(10.7), Inches(0.9),
          "A Real-Time Computer Vision Framework for Traffic Accident Detection\n"
          "and Automated Alerting using YOLOv8 and DeepSORT",
          size=17, color=WHITE, font=FONT_BODY, align=PP_ALIGN.CENTER, line_spacing=1.2)

add_text(s, Inches(1.0), Inches(4.85), Inches(11.3), Inches(0.4),
          "Project Phase-II (21AIP81)", size=14, color=AMBER, bold=True, font=FONT_MONO,
          align=PP_ALIGN.CENTER)

add_text(s, Inches(1.0), Inches(5.45), Inches(11.3), Inches(0.35),
          "Presented by", size=11, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.0), Inches(5.75), Inches(11.3), Inches(0.4),
          "Pratham M S (22SEAI46)   ·   Sumukh R Aaradhya (22SEAI62)   ·   Syed Shafaq Ahmed (22SEAI64)",
          size=13, color=WHITE, font=FONT_BODY, align=PP_ALIGN.CENTER)

add_text(s, Inches(1.0), Inches(6.35), Inches(11.3), Inches(0.35),
          "Guide: Meghana, Assistant Professor, B5", size=12, color=GREY, font=FONT_BODY,
          align=PP_ALIGN.CENTER)
add_text(s, Inches(1.0), Inches(6.68), Inches(11.3), Inches(0.35),
          "Dept. of AI & ML, MUSE, UoM, Mysuru-06", size=11, color=DGREY, font=FONT_MONO,
          align=PP_ALIGN.CENTER)

# ============================================================ SLIDE 2 -- Contents
s = new_slide()
header(s, "Presentation Roadmap", "Contents", number=2)
left_items = [
    "01   Introduction",
    "02   Literature Review",
    "03   Research Gap",
    "04   Problem Statement",
    "05   Objectives",
    "06   Proposed Methodology",
    "07   System Architecture",
    "08   Implementation Details",
    "09   Objectives 1 & 2 -- Analysis, Design & Model Development",
]
right_items = [
    "10   Objectives 3 & 4 -- Implementation, Testing & Evaluation",
    "11   Experimental Results",
    "12   Result Analysis",
    "13   Screenshots / Output Samples",
    "14   Conclusion",
    "15   Future Work",
    "16   References",
    "17   Thank You",
    "18   Coding Demonstration",
]
add_bullets(s, Inches(0.7), Inches(1.85), Inches(6.0), Inches(5.2), left_items, size=15.5,
            space_after=16)
add_bullets(s, Inches(6.9), Inches(1.85), Inches(6.0), Inches(5.2), right_items, size=15.5,
            space_after=16)
footer(s)

# ============================================================ SLIDE 3 -- Introduction
s = new_slide()
header(s, "Chapter 1", "Introduction", number=3)
items = [
    "Road traffic collisions are one of the leading causes of preventable injury and death worldwide, and a large share go undetected until reported after the fact.",
    "CORVUS identifies vehicle-vehicle and vehicle-pedestrian collisions directly from monocular CCTV footage, in real time, without radar, lidar, or inductive-loop sensors.",
    "Existing fixed-infrastructure systems (radar, inductive loops) are hardware-dependent and expensive to deploy; existing vision-only systems (CNN classifiers, anomaly-score models) rely on a single cue and do not fuse multiple signals.",
    "A single detection cue is unreliable on its own: bounding-box overlap alone fires on any two vehicles passing close at an oblique angle; sudden deceleration alone fires on every vehicle stopping at a red light.",
    "Scope: real-time detection and severity scoring from a single fixed camera feed, per-site zone calibration, automatic evidentiary output, dual-channel alerting, and a deployment path scaling from GPU workstation to CPU-only edge hardware.",
    "Out of scope: multi-camera fusion, true 3-D depth estimation, predicting a collision before contact, and direct emergency-dispatch integration -- each identified as future work.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=16.5,
            space_after=14, line_spacing=1.2)
footer(s)

# ============================================================ SLIDE 4 -- Literature Review
s = new_slide()
header(s, "Chapter 2", "Literature Review", number=4)
items = [
    "YOLO family & instance segmentation -- single-pass detection (Redmon et al. [1]) and Ultralytics YOLOv8 [4] give CORVUS pixel-accurate mask contours instead of coarse boxes.",
    "Multi-object tracking -- SORT [5] paired a Kalman filter with Hungarian assignment; DeepSORT [6] added an appearance embedding for persistent identity through occlusion.",
    "Time-to-collision & kinematic prediction -- classical TTC formulations [8], [9] inform CORVUS's continuous risk-gauge computation, separate from the discrete accident vote.",
    "Zone / polygon-based scene reasoning -- rule-based, site-calibrated polygon layers tie detector/tracker output to spatially meaningful regions (intersection, crosswalk, no-entry).",
    "OpenVINO & edge-AI optimisation [10] -- quantised inference paths enable CPU-class deployment without a GPU workstation.",
    "Existing accident-detection systems [11], [12], [28], [29] -- CNN clip classifiers and anomaly-score models each rely on a single decision signal, with no multi-signal fusion and no edge-deployment path.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=16,
            space_after=13, line_spacing=1.18)
footer(s)

# ============================================================ SLIDE 5 -- Research Gap
s = new_slide()
header(s, "Comparative Analysis", "Research Gap", number=5)

headers = ["System", "Real-Time", "Edge-Deployable", "Multi-Signal", "Accuracy"]
rows = [
    ["Ijjina et al. -- CNN clip classifier", "Partial", "No", "No", "Not multi-signal"],
    ["Singh & Mohan -- stacked autoencoder", "Partial", "No", "No (anomaly score)", "Anomaly-score based"],
    ["Radar / inductive-loop ITS", "Yes", "N/A (fixed hardware)", "No", "Hardware-dependent"],
    ["Chan et al. -- DSA-RNN (dashcam)", "Yes (anticipatory)", "No", "No (single RNN score)", "Anticipation-time metric"],
    ["Yao et al. -- DoTA (unsupervised)", "Partial", "No", "No (anomaly score)", "AUC-based"],
    ["CORVUS (this work)", "Yes (~8-30 FPS, GPU)", "Yes (OpenVINO INT8)", "Yes (up to 7 signals, >=2 req.)", "~91% (90-92%)"],
]
tx, ty, tw, th = Inches(0.55), Inches(1.75), Inches(12.25), Inches(4.6)
gtbl = s.shapes.add_table(len(rows)+1, len(headers), tx, ty, tw, th).table
gtbl.columns[0].width = Inches(3.6)
for i in range(1, 5):
    gtbl.columns[i].width = Inches((12.25 - 3.6) / 4)
for c, htext in enumerate(headers):
    cell = gtbl.cell(0, c)
    cell.text = htext
    cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0,0,0)
    p = cell.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(13); p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = WHITE; p.runs[0].font.name = FONT_MONO
for r, row in enumerate(rows, start=1):
    for c, val in enumerate(row):
        cell = gtbl.cell(r, c)
        cell.text = val
        is_corvus = (r == len(rows))
        cell.fill.solid()
        cell.fill.fore_color.rgb = RED_DIM if is_corvus else PANEL
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(12)
        p.runs[0].font.color.rgb = WHITE if is_corvus else GREY
        p.runs[0].font.bold = is_corvus
        p.runs[0].font.name = FONT_BODY
add_text(s, Inches(0.55), Inches(6.55), Inches(12.0), Inches(0.6),
          "No existing system combines real-time inference, edge-deployability, and multi-signal false-positive suppression in one framework -- CORVUS fills that gap.",
          size=14.5, color=AMBER, italic=True, font=FONT_BODY)
footer(s)

# ============================================================ SLIDE 6 -- Problem Statement
s = new_slide()
header(s, "Chapter 1.3", "Problem Statement", number=6)
items = [
    "Automated accident detection from video is difficult because a single cue is rarely reliable on its own.",
    "Bounding-box overlap alone fires on any two vehicles that merely pass close to the camera at an oblique angle.",
    "Sudden deceleration alone fires on every vehicle stopping at a red light.",
    "A single global threshold tuned on one video typically breaks down on the next.",
    "A practical detector must fuse several independent, individually weak signals into one confident decision.",
    "It must keep working across scene cuts and viewpoint changes in real-world footage.",
    "It must run at usable frame rates on hardware a municipal or campus deployment can actually afford -- including a CPU-only edge device in the field, not just a workstation GPU in a lab.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=17,
            space_after=15, line_spacing=1.2)
footer(s)

# ============================================================ SLIDE 7 -- Objectives
s = new_slide()
header(s, "Chapter 1.4", "Objectives", number=7)
objs = [
    ("O1", "Real-time, vision-only collision detection",
     "Detect vehicle-vehicle and vehicle-pedestrian collisions in real time from ordinary monocular CCTV/traffic-camera video, with no additional sensors."),
    ("O2", "Multi-signal fusion for false-positive suppression",
     "Fuse up to seven independent kinematic and geometric signals so no single noisy cue can trigger a false alert -- a minimum of two concurring signals is always required."),
    ("O3", "Edge-deployable, hardware-scalable architecture",
     "Run the same codebase from a GPU workstation (TensorRT FP16 segmentation) down to CPU-only edge hardware (OpenVINO INT8, box-only fallback mode)."),
    ("O4", "Automatic incident documentation and live monitoring",
     "Produce a complete, hands-off audit trail for every confirmed event -- a replay clip, a structured JSONL log record, a timestamped screenshot, and a live HUD."),
]
y = Inches(1.75)
for tag, title, desc in objs:
    add_rect(s, Inches(0.7), y + Inches(0.05), Inches(0.85), Inches(0.85), RED_DIM)
    add_text(s, Inches(0.7), y + Inches(0.05), Inches(0.85), Inches(0.85), tag, size=22,
              color=RED, bold=True, font=FONT_MONO, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(1.75), y, Inches(10.9), Inches(0.4), title, size=17, color=WHITE,
              bold=True, font=FONT_HEAD)
    add_text(s, Inches(1.75), y + Inches(0.42), Inches(10.9), Inches(0.65), desc, size=13.5,
              color=GREY, font=FONT_BODY, line_spacing=1.15)
    y += Inches(1.3)
footer(s)

# ============================================================ SLIDE 8 -- Proposed Methodology
s = new_slide()
header(s, "Approach", "Proposed Methodology", number=8)
items = [
    "A YOLOv8 segmentation detector produces pixel-accurate mask contours for every vehicle and pedestrian in frame, not just bounding boxes.",
    "A DeepSORT multi-object tracker assigns persistent track IDs and derives per-frame velocity, speed, and heading for every tracked object.",
    "A rule-based zone-reasoning layer independently enforces intrusion, loitering, and speed rules per calibrated polygon (Intersection / Crosswalk / No-Entry).",
    "A collision inference engine fuses up to seven independent kinematic and geometric signals -- mask overlap, sudden stop, direction change, closing velocity, speed differential, sustained overlap, and sudden-contact -- requiring at least two concurring signals before confirming an event.",
    "A continuous time-to-collision (TTC) computation drives the HUD's live threat gauge, independent of the discrete accident vote.",
    "Every confirmed event automatically produces a replay clip, a structured log entry, a screenshot, and simultaneous Discord + Telegram alerts.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=16,
            space_after=13, line_spacing=1.18)
footer(s)

# ============================================================ SLIDE 9 -- System Architecture
s = new_slide()
header(s, "Chapter 3", "System Architecture -- Workflow + Module Diagram", number=9)
pic = s.shapes.add_picture(IMG_ARCH, Inches(0.66), Inches(1.75), height=Inches(5.35))
pic.left = int((SLIDE_W - pic.width) / 2)

# ============================================================ SLIDE 10 -- Implementation Details
s = new_slide()
header(s, "Chapter 4", "Implementation Details", number=10)
headers = ["Layer", "Technology", "Role in CORVUS"]
rows = [
    ["Language / runtime", "Python 3.13", "All modules"],
    ["Detection", "Ultralytics YOLOv8 (yolov8s-seg / yolov8n)", "Vehicle + pedestrian detection, segmentation"],
    ["Detection acceleration", "TensorRT FP16 (.engine) / OpenVINO INT8", "GPU speed-up / CPU edge deployment"],
    ["Tracking", "deep-sort-realtime (DeepSORT)", "Persistent track ID + Re-ID under occlusion"],
    ["Tracking fallback", "SciPy (linear_sum_assignment)", "Hungarian centroid tracker, no GPU required"],
    ["Computer vision", "OpenCV", "Frame I/O, polygon tests, mask rasterisation, HUD"],
    ["Alerting", "Requests + python-telegram Bot API", "Discord webhook + Telegram bot dispatch"],
]
tx, ty, tw, th = Inches(0.55), Inches(1.7), Inches(12.25), Inches(3.55)
gtbl = s.shapes.add_table(len(rows)+1, len(headers), tx, ty, tw, th).table
gtbl.columns[0].width = Inches(2.6); gtbl.columns[1].width = Inches(4.85); gtbl.columns[2].width = Inches(4.8)
for c, htext in enumerate(headers):
    cell = gtbl.cell(0, c); cell.text = htext
    cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0,0,0)
    p = cell.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(13); p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = WHITE; p.runs[0].font.name = FONT_MONO
for r, row in enumerate(rows, start=1):
    for c, val in enumerate(row):
        cell = gtbl.cell(r, c); cell.text = val
        cell.fill.solid(); cell.fill.fore_color.rgb = PANEL
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(11.5); p.runs[0].font.color.rgb = GREY; p.runs[0].font.name = FONT_BODY
add_bullets(s, Inches(0.55), Inches(5.45), Inches(12.25), Inches(1.4), [
    "Nine cooperating modules, each single-responsibility, threaded together per frame by one orchestrator (main.py).",
    "config.py centralises every threshold; edge_config.py switches deployment mode (--edge: box-only, no DeepSORT, lower display scale).",
], size=14, space_after=8)
footer(s)

# ============================================================ SLIDE 11 -- Objectives 1&2 split
s = new_slide()
header(s, "Execution Plan", "Objectives 1 & 2: Analysis, Design & Model Development", number=11)
add_rect(s, Inches(6.55), Inches(1.7), Pt(2), Inches(5.1), RED_DIM)
add_text(s, Inches(0.7), Inches(1.65), Inches(5.6), Inches(0.4), "ANALYSIS & DESIGN",
          size=15, color=RED, bold=True, font=FONT_MONO)
add_bullets(s, Inches(0.7), Inches(2.15), Inches(5.6), Inches(4.6), [
    "Signal-independence philosophy: every module produces a self-contained, inspectable output that the next stage consumes.",
    "Nine single-responsibility modules mapped directly onto main.py's own import list -- no monolithic end-to-end model.",
    "Per-site zone calibration via an interactive click-to-draw tool (calibrate_zones.py) -- Intersection, Crosswalk, No-Entry polygons.",
    "End-to-end data flow designed so any one signal's false-positive behaviour can be diagnosed and re-tuned in isolation.",
], size=13.5, space_after=12, line_spacing=1.15)
add_text(s, Inches(6.85), Inches(1.65), Inches(5.8), Inches(0.4), "MODEL / ALGORITHM CHOICE",
          size=15, color=RED, bold=True, font=FONT_MONO)
add_bullets(s, Inches(6.85), Inches(2.15), Inches(5.8), Inches(4.6), [
    "YOLOv8 segmentation chosen over box-only detection: pixel-accurate mask contours let the collision engine test true silhouette overlap, directly mitigating 2-D projection ambiguity.",
    "DeepSORT chosen over plain SORT: an appearance embedding meaningfully improves ID persistence through brief occlusion, at the cost of being GPU-dependent.",
    "TensorRT FP16 export for GPU workstations; OpenVINO INT8 export for CPU-only edge hardware -- same codebase, same collision logic, two acceleration paths.",
    "Multi-signal voting (>=2 of 7) chosen over any single threshold, because live-testing showed every individual signal has a real, distinct false-positive pattern.",
], size=13.5, space_after=12, line_spacing=1.15)
footer(s)

# ============================================================ SLIDE 12 -- Objectives 3&4 split
s = new_slide()
header(s, "Execution Plan", "Objectives 3 & 4: Implementation, Testing & Performance Evaluation", number=12)
add_rect(s, Inches(6.55), Inches(1.7), Pt(2), Inches(5.1), RED_DIM)
add_text(s, Inches(0.7), Inches(1.65), Inches(5.6), Inches(0.4), "CODING, INTEGRATION & TESTING",
          size=15, color=RED, bold=True, font=FONT_MONO)
add_bullets(s, Inches(0.7), Inches(2.15), Inches(5.6), Inches(4.6), [
    "Five successive rounds of live-testing against recorded traffic footage, each round fixing one observed false-positive pattern.",
    "Every non-obvious threshold in config.py carries an inline record of the false-positive pattern it was introduced to fix.",
    "Scene-cut resilience validated directly: all 5 hard cuts in the test session were detected and correctly triggered a tracker/confirmation-state reset.",
    "Alert delivery verified end-to-end: a real confirmed event was allowed to trigger a live alert to both Discord and Telegram simultaneously.",
], size=13.5, space_after=12, line_spacing=1.15)
add_text(s, Inches(6.85), Inches(1.65), Inches(5.8), Inches(0.4), "EVALUATION APPROACH & COMPARISON",
          size=15, color=RED, bold=True, font=FONT_MONO)
add_bullets(s, Inches(6.85), Inches(2.15), Inches(5.8), Inches(4.6), [
    "Frame-by-frame verification against a manually reviewed ground-truth clip: 5 of 6 real collisions correctly detected.",
    "Validation review across the incident archive: ~91% detection accuracy (90-92%).",
    "Compared against radar/inductive-loop systems (hardware-dependent, no multi-signal) and CNN/anomaly-score systems (single-cue, no edge path) -- CORVUS is the only one combining all three.",
    "Segmentation vs. box-only trade-off measured directly: mask-accurate overlap costs FPS but is what makes multi-signal fusion reliable.",
], size=13.5, space_after=12, line_spacing=1.15)
footer(s)

# ============================================================ SLIDE 13 -- Experimental Results
s = new_slide()
header(s, "Chapter 5", "Experimental Results", number=13)
kpi_block(s, Inches(0.6), Inches(1.85), Inches(2.85), "547 / 547", "Frames Processed")
kpi_block(s, Inches(3.55), Inches(1.85), Inches(2.85), "7", "Accident Events")
kpi_block(s, Inches(6.5), Inches(1.85), Inches(2.85), "4 / 3", "High / Medium Severity")
kpi_block(s, Inches(9.45), Inches(1.85), Inches(2.85), "~91%", "Detection Accuracy")

chart_data = CategoryChartData()
chart_data.categories = ["BOX_OVERLAP", "SUDDEN_CONTACT", "SUSTAINED_OVERLAP"]
chart_data.add_series("Times triggered", (5, 4, 3))
gframe = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.7), Inches(3.35),
                              Inches(6.0), Inches(3.3), chart_data)
chart = gframe.chart
chart.has_legend = False
chart.has_title = False
plot = chart.plots[0]
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = RED
cat_ax = chart.category_axis
cat_ax.tick_labels.font.size = Pt(12); cat_ax.tick_labels.font.color.rgb = WHITE
cat_ax.tick_labels.font.name = FONT_MONO
cat_ax.format.line.color.rgb = DGREY
val_ax = chart.value_axis
val_ax.tick_labels.font.size = Pt(11); val_ax.tick_labels.font.color.rgb = GREY
val_ax.format.line.color.rgb = DGREY
val_ax.major_gridlines.format.line.color.rgb = RGBColor(0x22, 0x22, 0x26)
chart.font.name = FONT_BODY

chart_data2 = CategoryChartData()
chart_data2.categories = ["HIGH", "MEDIUM", "LOW"]
chart_data2.add_series("Severity", (4, 3, 0))
gframe2 = s.shapes.add_chart(XL_CHART_TYPE.PIE, Inches(7.05), Inches(3.35),
                               Inches(5.6), Inches(3.3), chart_data2)
chart2 = gframe2.chart
chart2.has_title = False
chart2.has_legend = True
chart2.legend.position = XL_LEGEND_POSITION.RIGHT
chart2.legend.font.size = Pt(12); chart2.legend.font.color.rgb = WHITE
chart2.legend.font.name = FONT_MONO
chart2.legend.include_in_layout = False
plot2 = chart2.plots[0]
plot2.has_data_labels = True
plot2.data_labels.font.size = Pt(12); plot2.data_labels.font.color.rgb = WHITE
sev_colors = [RED, AMBER, GREEN]
for i, point in enumerate(plot2.series[0].points):
    point.format.fill.solid()
    point.format.fill.fore_color.rgb = sev_colors[i]
add_text(s, Inches(0.7), Inches(6.75), Inches(6.0), Inches(0.35), "Signal contributions (video-1 session)",
          size=11.5, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(7.05), Inches(6.75), Inches(5.6), Inches(0.35), "Severity breakdown (7 confirmed events)",
          size=11.5, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
footer(s)

# ============================================================ SLIDE 14 -- Result Analysis
s = new_slide()
header(s, "Interpretation", "Result Analysis", number=14)
items = [
    "The >=2-signal requirement is doing real work: BOX_OVERLAP never fired alone across the confirmed events -- exactly as MIN_SIGNALS_REQUIRED is designed to enforce.",
    "SUDDEN_CONTACT (brief, low-speed impacts) and BOX_OVERLAP (sustained mask overlap) were the two dominant contributing signals in this session.",
    "All 4 HIGH-severity events combined multiple concurring signals, consistent with the weighted severity model (score = min(1, sum of fired-signal weights)).",
    "Both Discord and Telegram alerts fired successfully for every confirmed event, with a screenshot, severity, and trigger flags attached -- confirming the dual-channel alert path is failure-isolated per channel.",
    "Detection accuracy (~91%, 90-92% across the incident archive) is achieved without any single global threshold -- every threshold is signal-specific and independently tunable.",
    "Remaining false positives trace to monocular depth ambiguity: two vehicles in adjacent lanes can appear to touch from an oblique camera angle without being physically close.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=15.5,
            space_after=13, line_spacing=1.18)
footer(s)

# ============================================================ SLIDE 15 -- Screenshots
s = new_slide()
header(s, "Live System", "Screenshots / Output Samples", number=15)
picture_framed(s, IMG_NORMAL, Inches(0.55), Inches(1.68), Inches(6.05), Inches(2.55))
picture_framed(s, IMG_ACCIDENT, Inches(6.75), Inches(1.68), Inches(6.05), Inches(2.55))
add_text(s, Inches(0.55), Inches(4.28), Inches(6.05), Inches(0.3), "Normal monitoring -- full HUD",
          size=11.5, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(6.75), Inches(4.28), Inches(6.05), Inches(0.3), "Confirmed collision -- HUD banner + threat gauge",
          size=11.5, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
picture_framed(s, IMG_DISCORD, Inches(2.9), Inches(4.68), Inches(2.5), Inches(2.35), border=AMBER)
picture_framed(s, IMG_TELEGRAM, Inches(7.9), Inches(4.68), Inches(2.5), Inches(2.35), border=AMBER)
add_text(s, Inches(2.9), Inches(7.05), Inches(2.5), Inches(0.3), "Discord #system-alerts",
          size=10.5, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(7.9), Inches(7.05), Inches(2.5), Inches(0.3), "Telegram bot",
          size=10.5, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)

# ============================================================ SLIDE 16 -- Conclusion
s = new_slide()
header(s, "Chapter 6.1", "Conclusion", number=16)
items = [
    "CORVUS demonstrates that a purely vision-based, multi-signal fusion approach can detect vehicle-vehicle and vehicle-pedestrian collisions from ordinary monocular traffic-camera footage in real time, without additional sensing hardware.",
    "Achieves approximately 91% detection accuracy across its validation review.",
    "Its nine-module architecture keeps detection, tracking, zone reasoning, behaviour classification, and collision inference as independently inspectable stages.",
    "On its most recent recorded evaluation session, the system processed a 2,088-frame video end to end, correctly handled five hard scene cuts, and confirmed 14 collision events with a severity-weighted, multi-signal vote.",
    "All four core objectives -- real-time vision-only detection, multi-signal false-positive suppression, edge-deployable architecture, and automatic incident documentation -- have been met in the current build.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=17,
            space_after=16, line_spacing=1.2)
footer(s)

# ============================================================ SLIDE 17 -- Future Work
s = new_slide()
header(s, "Chapter 6.3", "Future Work", number=17)
items = [
    "Broader ground-truth benchmark dataset -- extend the hand-reviewed incident archive across a wider range of footage, camera geometries, and conditions.",
    "Multi-camera fusion -- correlate detections across overlapping or adjacent camera views for the same intersection to reduce single-viewpoint occlusion and depth ambiguity.",
    "Stereo or learned depth estimation -- introduce a depth cue as an additional collision signal, to directly address 2-D projection ambiguity rather than only mitigating it.",
    "On-device edge benchmarking -- run the existing OpenVINO INT8 export path on representative edge hardware (Raspberry Pi / Jetson-class) to measure throughput and accuracy directly.",
    "Additional alert channels -- SMS, email, or direct emergency-dispatch API integration, matching the smart-city integration scenario.",
    "Sudden-contact signal is already active in the current build, catching brief, low-speed impacts too fast for the sustained multi-signal gate -- continued validation against a wider range of low-speed-impact footage is ongoing.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), items, size=15.5,
            space_after=13, line_spacing=1.18)
footer(s)

# ============================================================ SLIDE 18 -- References
s = new_slide()
header(s, "Bibliography", "References", number=18)
refs = [
    "[1]  J. Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” CVPR, 2016.",
    "[4]  G. Jocher et al., “Ultralytics YOLOv8,” 2023.",
    "[5]  A. Bewley et al., “Simple Online and Realtime Tracking,” ICIP, 2016.",
    "[6]  N. Wojke et al., “Simple Online and Realtime Tracking with a Deep Association Metric,” ICIP, 2017.",
    "[9]  M. M. Minderhoud and P. H. L. Bovy, “Extended Time-to-Collision Measures for Road Traffic Safety Assessment,” Accident Analysis & Prevention, 2001.",
    "[10]  Intel Corporation, “OpenVINO Toolkit Documentation.”",
    "[11]  E. P. Ijjina et al., “Computer Vision-based Accident Detection in Traffic Surveillance,” ICCCNT, 2019.",
    "[19]  NVIDIA Corporation, “TensorRT Developer Guide.”",
    "[20]  Discord Inc., “Discord Developer Documentation -- Webhook Resource.”",
    "[21]  Telegram FZ-LLC, “Telegram Bot API Documentation.”",
    "[22]  World Health Organization, “Global Status Report on Road Safety 2023,” Geneva, 2023.",
]
add_bullets(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2), refs, size=13.5,
            space_after=10, line_spacing=1.1, bullet_color=DGREY)
footer(s)

# ============================================================ SLIDE 19 -- Thank You
s = new_slide()
add_rect(s, 0, 0, SLIDE_W, Inches(0.12), RED)
add_rect(s, 0, Inches(7.38), SLIDE_W, Inches(0.12), RED)
add_text(s, Inches(1.0), Inches(2.7), Inches(11.3), Inches(1.2),
          "THANK YOU", size=56, color=RED, bold=True, font=FONT_HEAD, align=PP_ALIGN.CENTER)
add_rect(s, Inches(5.2), Inches(3.85), Inches(2.9), Pt(1.6), RED)
add_text(s, Inches(1.0), Inches(4.1), Inches(11.3), Inches(0.5),
          "CORVUS -- Collision Onset Recognition · Vehicular · Unified · Surveillance",
          size=14, color=GREY, font=FONT_MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.0), Inches(5.0), Inches(11.3), Inches(0.4),
          "Questions & Discussion", size=16, color=WHITE, font=FONT_BODY, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.0), Inches(6.55), Inches(11.3), Inches(0.35),
          "Pratham M S  ·  Sumukh R Aaradhya  ·  Syed Shafaq Ahmed  --  Dept. of AI & ML, MUSE, UoM",
          size=11, color=DGREY, font=FONT_MONO, align=PP_ALIGN.CENTER)

# ============================================================ SLIDE 20 -- Coding Demonstration
s = new_slide()
header(s, "Live Demo", "Coding Demonstration", number=20)
items = [
    "Live run: python main.py --video videos/test_video2.mp4 (also test_video.mp4)",
    "TensorRT FP16 segmentation engine, DeepSORT tracking, 3 calibrated zones loaded at startup.",
    "Controls: Q = quit,  SPACE = pause,  S = screenshot,  Z = zones overlay,  H = speed heatmap.",
    "Watch for: corner-bracket tracking reticles, live threat-level gauge, zone dwell bars, and the red COLLISION DETECTED banner on a confirmed event.",
    "On confirmation: automatic screenshot, replay clip, JSONL log entry, and simultaneous Discord + Telegram alerts -- all visible live in this session.",
]
add_bullets(s, Inches(0.7), Inches(1.9), Inches(11.9), Inches(4.8), items, size=17,
            space_after=18, line_spacing=1.25)
footer(s)

prs.save(ROOT + r"\CORVUS_Presentation.pptx")
print("stage 1 saved, slide count:", len(prs.slides._sldIdLst))
