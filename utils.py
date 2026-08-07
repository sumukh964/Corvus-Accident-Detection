# =============================================================================
# utils.py — CORVUS Visual Engine v4.0
# =============================================================================
# NEW FEATURES:
#   - Corner bracket targeting reticles  (replaces plain rectangles)
#   - Speed streak lines                 (replaces motion trail dots)
#   - Speed in km/h on labels
#   - Detection confidence arc           (partial circle around centroid)
#   - Behavior tags  AGGRESSIVE / ERRATIC / STATIONARY
#   - Pulse rings on sudden acceleration
#   - Speed heatmap overlay              (H key toggle)
#   - Trajectory prediction lines + red X at path convergence
#   - Near-miss highlights + cumulative counter
#   - Shockwave rings on collision
#   - White flash + freeze-frame effect
#   - Live risk graph in side panel      (filled sparkline)
#   - Zone dwell fill bars inside polygon
# =============================================================================

import os
import time
import math
import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)

# ── Color Palette — futuristic red/black reskin ──────────────────────────────
#   Former cyan-family chrome (title, borders, ticks, badges, grid, radar) is
#   now red. Severity semantics (GREEN=safe/AMBER=warn/CRIMSON=danger) are kept
#   as-is — they encode meaning, not decoration, and CRIMSON already reads as
#   this theme's "hot" alert red, so real accidents still pop against the
#   ambient red chrome.
C_VOID         = ( 10,   6,   3)   # #03060a  --void
C_PANEL_BG     = (  6,   6,   6)   # right sidebar background — near-black
# ambient red @ 18% over panel bg, pre-blended
C_LINE         = ( 44,  12,  16)
C_LINE_FULL    = ( 45,  40, 230)   # ambient red, full-strength border/grid/title
C_CYAN         = ( 45,  40, 230)   # primary chrome red (was cyan #2fe6ff)
C_CYAN_DIM     = ( 45,  40, 230)   # same hue, used at low alpha
C_VIOLET       = (235, 235, 235)   # neutral white/grey (was violet) — pedestrians, subtitle
C_AMBER        = ( 32, 176, 255)   # #ffb020  --amber
C_CRIMSON      = ( 92,  59, 255)   # #ff3b5c  --crimson (kept as the "hot" alert red)
C_MIST         = (235, 235, 235)   # near-white — body/label text (dimmer than C_TEXT only
                                   # for a subtle label-vs-value hierarchy, not color)
C_TEXT         = (255, 255, 255)   # pure white — body/value text. Per design direction,
                                   # ALL body/content text is white; red/orange is reserved
                                   # for headers, dividers, alerts, and the wordmark only.
C_GREEN        = (176, 255,  77)   # #4dffb0  --green (still used: near-miss/zone-clear dots)
C_PEDESTRIAN   = (220, 190, 150)   # warm pedestrian (unchanged)
# Threat-level gauge — orange (low risk) -> red (high risk) only, no green/amber.
# Scoped to the gauge component; severity colors elsewhere (incident log, alerts) unchanged.
C_GAUGE_LOW    = ( 26, 122, 255)   # #ff7a1a orange
C_GAUGE_HIGH   = ( 45,  35, 235)   # #eb2d2d deep red
C_BLACK_BORDER = ( 26,  26,  26)   # stationary-vehicle border — near-black, kept
                                   # slightly above pure 0 so it stays visible
                                   # against dark asphalt/shadow in the footage
# ── Aliases ──────────────────────────────────────────────────────────────────
C_TEXT_PRIMARY = C_TEXT
C_TEXT_DIM     = C_MIST
C_DIVIDER      = C_LINE
C_RED          = C_CRIMSON
C_RED_DARK     = ( 45,   5,  15)
C_PANEL_BORDER = C_LINE_FULL
C_CORVUS_TITLE = C_CYAN

# ── PIL font helpers — JetBrains Mono + Orbitron variable fonts ───────────────
_FONT_JB  = "assets/JetBrains_Mono/JetBrainsMono-VariableFont_wght.ttf"
_FONT_ORB = "assets/Orbitron/Orbitron-VariableFont_wght.ttf"
_pil_font_cache: dict = {}          # (face, size_px, weight) → ImageFont or False
_PIL_SIZES = {'xs': 9, 'sm': 11, 'med': 13, 'lg': 18, 'xl': 22, 'xxl': 34}


def _get_pil_font(size_px, face="jb", weight=400):
    """Load and cache a variable font at size_px/weight. Returns font or False."""
    key = (face, size_px, weight)
    if key not in _pil_font_cache:
        try:
            from PIL import ImageFont
            path = _FONT_ORB if face == "orb" else _FONT_JB
            font = ImageFont.truetype(path, size_px)
            font.set_variation_by_axes([weight])
            _pil_font_cache[key] = font
        except Exception:
            _pil_font_cache[key] = False
    return _pil_font_cache[key]


def _pil_fonts():
    """Legacy helper used by draw_track — returns (sm, med) JetBrains Mono."""
    return _get_pil_font(11, "jb", 400), _get_pil_font(13, "jb", 400)


_pil_font_sm  = None   # kept for legacy compat
_pil_font_med = None


def _pil_measure(text, size="sm", face="jb", weight=None):
    """Return (width, height) of rendered text, or (0, 0) on failure."""
    px_sz = _PIL_SIZES.get(size, 11) if isinstance(size, str) else int(size)
    wt    = weight if weight is not None else (700 if face == "orb" else 400)
    font  = _get_pil_font(px_sz, face=face, weight=wt)
    if not font:
        return 0, 0
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        return len(text) * px_sz // 2, px_sz


def _pil_label(canvas, text, x, y, color, size="sm", face="jb", weight=None):
    """Draw text via PIL onto the BGR canvas.
    x = left edge of ink. y = bottom edge of ink.
    face: 'jb'=JetBrains Mono (default), 'orb'=Orbitron
    weight: 100-900; defaults to 400 for jb, 700 for orb"""
    px_sz = _PIL_SIZES.get(size, 11) if isinstance(size, str) else int(size)
    wt    = weight if weight is not None else (700 if face == "orb" else 400)
    font  = _get_pil_font(px_sz, face=face, weight=wt)
    if not font:
        cv_sc = {9:0.28,11:0.34,13:0.40,18:0.52,22:0.64,34:0.95}.get(px_sz, 0.34)
        cv2.putText(canvas, text.replace('\xb7','-').replace('·','-'),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, cv_sc, color, 1, cv2.LINE_AA)
        return
    from PIL import Image, ImageDraw
    try:
        bbox = font.getbbox(text)
    except AttributeError:
        tw2, th2 = font.getsize(text)
        bbox = (0, 0, tw2, th2)
    x0, y0, x1, y1 = bbox
    tw, th = x1 - x0, y1 - y0
    if tw <= 0 or th <= 0:
        return
    # Render onto scratch with origin at (2,2) — large enough to hold full glyph
    s_w = max(1, x1 + 4)
    s_h = max(1, y1 + 4)
    scratch = Image.new("RGB", (s_w, s_h), (0, 0, 0))
    ImageDraw.Draw(scratch).text((2, 2), text, font=font,
                                 fill=(color[2], color[1], color[0]))
    arr = np.array(scratch)[:, :, ::-1]   # BGR
    # Crop to exact ink bounding box — eliminates all origin/bearing offsets
    rs = max(0, 2 + y0);  re = min(arr.shape[0], 2 + y1)
    cs = max(0, 2 + x0);  ce = min(arr.shape[1], 2 + x1)
    if re <= rs or ce <= cs:
        return
    ink    = arr[rs:re, cs:ce].astype(np.float32)
    ink_h, ink_w = ink.shape[:2]
    # place: x=left of ink, y=bottom of ink
    ys = max(0, y - ink_h);  ye = min(canvas.shape[0], y)
    xs = max(0, x);          xe = min(canvas.shape[1], x + ink_w)
    if ye <= ys or xe <= xs:
        return
    ay  = ys - (y - ink_h);  by_ = ay + (ye - ys)
    ax  = xs - x;             bx_ = ax + (xe - xs)
    region  = canvas[ys:ye, xs:xe].astype(np.float32)
    text_px = ink[ay:by_, ax:bx_]
    mask    = text_px.max(axis=2, keepdims=True) / 255.0
    canvas[ys:ye, xs:xe] = np.clip(region*(1-mask) + text_px*mask,
                                   0, 255).astype(np.uint8)


def _pil_label_glow(canvas, text, x, y, color, size_px=32, weight=900, glow_r=12):
    """Orbitron text with blurred glow halo. x=ink left, y=ink bottom (same convention as _pil_label)."""
    from PIL import Image, ImageDraw, ImageFilter
    font = _get_pil_font(size_px, face="orb", weight=weight)
    if not font:
        _pil_label(canvas, text, x, y, color, size=size_px, face="orb", weight=weight)
        return
    try:
        bbox = font.getbbox(text)
    except AttributeError:
        tw2, th2 = font.getsize(text)
        bbox = (0, 0, tw2, th2)
    x0, y0, x1, y1 = bbox
    tw, th = x1 - x0, y1 - y0
    if tw <= 0 or th <= 0:
        return
    # Scratch: font origin at (pad, pad); large enough for full glyph + glow bleed
    pad  = glow_r * 2 + 4
    s_w  = max(1, x1 + pad + glow_r)
    s_h  = max(1, y1 + pad + glow_r)
    rgb  = (color[2], color[1], color[0])
    glow  = Image.new("RGBA", (s_w, s_h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((pad, pad), text, font=font, fill=rgb + (210,))
    glow  = glow.filter(ImageFilter.GaussianBlur(glow_r))
    glow2 = Image.new("RGBA", (s_w, s_h), (0, 0, 0, 0))
    ImageDraw.Draw(glow2).text((pad, pad), text, font=font, fill=rgb + (180,))
    glow2 = glow2.filter(ImageFilter.GaussianBlur(glow_r // 2))
    sharp = Image.new("RGBA", (s_w, s_h), (0, 0, 0, 0))
    ImageDraw.Draw(sharp).text((pad, pad), text, font=font, fill=rgb + (255,))
    combined = Image.alpha_composite(Image.alpha_composite(glow, glow2), sharp)
    arr = np.array(combined)   # RGBA H×W×4
    # Ink bottom in scratch = pad+y1; ink left in scratch = pad+x0
    # Map so canvas (x, y) = ink (left, bottom)
    cy_off = y - (pad + y1)
    cx_off = x - (pad + x0)
    H2, W2 = arr.shape[:2]
    ys = max(0, cy_off);  ye = min(canvas.shape[0], cy_off + H2)
    xs = max(0, cx_off);  xe = min(canvas.shape[1], cx_off + W2)
    if ye <= ys or xe <= xs:
        return
    ay, by_ = ys - cy_off, ye - cy_off
    ax, bx_ = xs - cx_off, xe - cx_off
    region = canvas[ys:ye, xs:xe].astype(np.float32)
    fg     = arr[ay:by_, ax:bx_]
    fg_bgr = fg[:, :, :3][:, :, ::-1].astype(np.float32)
    alpha  = fg[:, :, 3:4].astype(np.float32) / 255.0
    canvas[ys:ye, xs:xe] = np.clip(region*(1-alpha)+fg_bgr*alpha, 0, 255).astype(np.uint8)

BEHAVIOR_COLOR = {
    "NORMAL"    : None,
    "AGGRESSIVE": C_AMBER,
    "ERRATIC"   : C_CRIMSON,
    "STATIONARY": C_VIOLET,
}
BEHAVIOR_TAG = {
    "AGGRESSIVE": "AGG",
    "ERRATIC"   : "ERT",
    "STATIONARY": "STA",
}

# ── Fixed per-track color palette (feature 3) ─────────────────────────────────
_TRACK_PALETTE = [
    (255, 230,  47),   # cyan
    (176, 255,  77),   # green
    ( 32, 176, 255),   # amber
    (255, 139, 154),   # violet
    (100, 220, 255),   # sky
    (200, 255, 100),   # lime
    (255, 100, 180),   # pink
    (120, 255, 200),   # aqua
]
_track_color_map = {}   # track_id → BGR color

def get_track_color(track_id, behavior="NORMAL", accident=False, speed=0.0):
    if accident:
        return _pulse_color(C_CRIMSON, 4.0)   # brighter pulsing red — stays distinct from ambient
    if speed > 1.5:
        return C_CYAN               # moving vehicle — ambient theme red
    return C_BLACK_BORDER           # stationary vehicle

# ── Layout — matches HTML grid (header 68 / footer 34 / sidebar 380) ─────────
# Sidebar widened from 320->380 to give the inner text/boxes room to be
# drawn larger and stay legible from a distance (projector/presentation).
CANVAS_W  = 1536
CANVAS_H  = 960
TOP_BAR_H = 68
BOT_BAR_H = 34
VIDEO_W   = CANVAS_W - 380          # 1156
VIDEO_H   = CANVAS_H - TOP_BAR_H - BOT_BAR_H   # 858
PANEL_X   = VIDEO_W
PANEL_W   = 380
PANEL_H   = CANVAS_H

# ── Pulse helpers ─────────────────────────────────────────────────────────────
def _pulse(hz=1.0, min_a=0.35, max_a=1.0):
    a = (math.sin(time.time() * hz * 2 * math.pi) + 1) / 2
    return min_a + a * (max_a - min_a)

def _pulse_color(color, hz=1.0):
    a = _pulse(hz)
    return tuple(int(c * a) for c in color)

def _gauge_color(frac):
    """Orange (low) -> red (high) interpolation for the threat gauge, frac in [0,1]."""
    frac = max(0.0, min(1.0, frac))
    return tuple(int(C_GAUGE_LOW[i] + (C_GAUGE_HIGH[i] - C_GAUGE_LOW[i]) * frac) for i in range(3))

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "fps"             : 0.0,
    "frame_no"        : 0,
    "n_vehicles"      : 0,
    "n_pedestrians"   : 0,
    "n_tracks"        : 0,
    "mode"            : "SEG",
    "session_start"   : time.time(),
    "active_zones"    : [],
    "event_log"       : [],
    "video_fps"       : 25.0,
    "pixels_per_meter": 8.0,
    "use_tensorrt"    : False,
    "near_miss_total" : 0,
    "_nm_active_pairs": set(),
    "risk_history"    : [],
    # feature 6 — TTC (folded into the Threat Level card as a conditional callout)
    "ttc_sec"         : None,
    "ttc_pair"        : None,
    "ttc_delta_kmh"   : 0.0,
    # zones — per-zone live occupancy for the ZONES card
    "zone_status"     : [],
    # feature 9 — filmstrip
    "accident_frames" : [],
    # feature 10 — heatmap toggle (UI state tracked in main; this is just side panel label)
    "heatmap_on"      : False,
    # Threat gauge needle animation (eased toward target, not snapped) and TTC
    # callout hysteresis (holds + fades instead of hard flickering in/out).
    "_gauge_angle"    : None,   # None = uninitialized, snaps to first real value
    "_ttc_anim"       : 0.0,
    "_ttc_last_seen"  : 0.0,
    "_ttc_last_val"   : None,
    "_ttc_last_pair"  : None,
}

def update_state(**kwargs):
    _state.update(kwargs)

def push_event(event):
    _state["event_log"].append({
        "ts"   : time.strftime("%H:%M:%S"),
        "ids"  : " ".join(str(i) for i in event.vehicle_ids),
        "flags": " + ".join(f[:4] for f in event.trigger_flags),
        "sev"  : event.severity,
        "score": event.severity_score,
        "epoch": time.time(),
    })
    if len(_state["event_log"]) > 8:
        _state["event_log"].pop(0)
    # Spike the risk graph on accident — already self-decaying (it just
    # appends into the rolling 120-frame history), so no separate persistent
    # "accident_active" flag is needed on top of it.
    hist = _state["risk_history"]
    hist.extend([1.0] * 6)
    if len(hist) > 120:
        _state["risk_history"] = hist[-120:]

# ── Animation state ───────────────────────────────────────────────────────────
_pulse_rings      = []
_collision_fx     = {"active": False, "cx": 0, "cy": 0, "t": 0}
_heatmap          = None
_track_prev_speed = {}
_vignette_cache   = {}        # (h,w) → precomputed alpha mask

# ── Depth helpers (feature 1) ─────────────────────────────────────────────────
def _depth_scale(cy, frame_h):
    """0.4 (top/far) → 1.0 (bottom/near) using y-position as depth proxy."""
    return 0.40 + 0.60 * min(1.0, cy / max(1, frame_h))

def _depth_fade(frame, x1, y1, x2, y2, alpha):
    """Blend a region of the frame towards black to simulate distance fade."""
    if alpha >= 0.99:
        return
    roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
    if roi.size == 0:
        return
    frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)] = (
        roi * alpha).astype(np.uint8)

# ── TTC computation (feature 6) ───────────────────────────────────────────────
def compute_ttc(tracks):
    """Return (ttc_sec, (id_a, id_b), delta_kmh) for closest approaching pair."""
    fps  = max(1.0, _state.get("video_fps", 25.0))
    ppm  = max(0.1, _state.get("pixels_per_meter", 8.0))
    best = (None, None, 0.0)
    active = [t for t in tracks if t.disappeared == 0 and t.speed > 0.3]
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            a, b = active[i], active[j]
            ax, ay = a.current_centroid()
            bx, by = b.current_centroid()
            dx, dy = bx-ax, by-ay
            dist   = math.hypot(dx, dy)
            if dist < 10:
                continue
            dvx = getattr(b,"vx",0) - getattr(a,"vx",0)
            dvy = getattr(b,"vy",0) - getattr(a,"vy",0)
            closing = -(dvx*dx + dvy*dy) / dist   # px/frame toward each other
            if closing > 0.2:
                ttc_f   = dist / closing
                ttc_s   = ttc_f / fps
                delta_k = closing / ppm * fps * 3.6
                if best[0] is None or ttc_s < best[0]:
                    best = (ttc_s, (a.id, b.id), delta_k)
    return best

# ── Heatmap ───────────────────────────────────────────────────────────────────
def update_heatmap(tracks, frame):
    """Accumulate per-vehicle speed blobs onto a heat map matching the frame size."""
    global _heatmap
    h, w = frame.shape[:2]
    if _heatmap is None or _heatmap.shape != (h, w):
        _heatmap = np.zeros((h, w), dtype=np.float32)

    _heatmap *= 0.96  # temporal decay

    for t in tracks:
        if t.disappeared > 0:
            continue
        cx  = max(0, min(w-1, t.current_centroid()[0]))
        cy  = max(0, min(h-1, t.current_centroid()[1]))
        spd = max(1.0, t.speed)
        r   = int(spd * 3 + 15)

        y1, y2 = max(0, cy-r), min(h, cy+r+1)
        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        if y2 <= y1 or x2 <= x1:
            continue

        gy = np.arange(y1, y2, dtype=np.float32) - cy
        gx = np.arange(x1, x2, dtype=np.float32) - cx
        GX, GY = np.meshgrid(gx, gy)
        sigma  = r * 0.45
        blob   = np.exp(-(GX**2 + GY**2) / (2.0 * sigma**2 + 1e-6)) * spd * 0.25
        _heatmap[y1:y2, x1:x2] += blob


def draw_heatmap_overlay(frame):
    """Render heat accumulator as a semi-transparent JET colormap layer."""
    if _heatmap is None:
        return
    h, w = frame.shape[:2]
    heat = cv2.resize(_heatmap, (w, h)) if _heatmap.shape != (h, w) else _heatmap
    mx   = heat.max()
    if mx < 0.5:
        return
    norm    = np.clip(heat / mx * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    alpha   = np.clip(norm.astype(np.float32) / 255.0 * 0.50, 0, 0.50)
    for c in range(3):
        frame[:, :, c] = np.clip(
            frame[:, :, c] * (1 - alpha) + colored[:, :, c] * alpha, 0, 255
        ).astype(np.uint8)


# ── Frame analytics ───────────────────────────────────────────────────────────
def compute_frame_data(tracks, near_miss_dist=80):
    """
    Compute per-frame analytics:
      near_miss_pairs: list of (id_a, id_b, dist, mid_x, mid_y)
      risk_score:      float 0..1
    Updates _state near_miss_total and risk_history.
    """
    active   = [t for t in tracks if t.disappeared == 0]
    vehicles = [t for t in active  if t.label != "person"]

    near_miss_pairs = []
    current_nm_keys = set()
    for i in range(len(vehicles)):
        for j in range(i+1, len(vehicles)):
            ta, tb = vehicles[i], vehicles[j]
            ax, ay = ta.current_centroid()
            bx, by = tb.current_centroid()
            dist   = math.hypot(ax-bx, ay-by)
            if 0 < dist < near_miss_dist:
                near_miss_pairs.append((ta.id, tb.id, dist, (ax+bx)//2, (ay+by)//2))
                current_nm_keys.add(frozenset({ta.id, tb.id}))

    prev_keys = _state.get("_nm_active_pairs", set())
    new_pairs = current_nm_keys - prev_keys
    _state["_nm_active_pairs"] = current_nm_keys
    if new_pairs:
        _state["near_miss_total"] = _state.get("near_miss_total", 0) + len(new_pairs)

    # Note: no persistent "accident just happened" term here by design —
    # push_event() already spikes risk_history directly to 1.0 for a few
    # frames on a real accident, and that's self-decaying (it ages out of
    # the rolling window naturally). A flag-based version of this used to
    # exist but never reset once set, permanently inflating every frame's
    # risk floor after the first accident of the session — removed.
    spd_sq = sum(t.speed**2 for t in vehicles)
    spd_risk = min(1.0, spd_sq / max(1, len(vehicles) * 400)) if vehicles else 0.0
    nm_risk  = min(1.0, len(near_miss_pairs) * 0.30)
    risk     = min(1.0, spd_risk * 0.50 + nm_risk * 0.50)

    hist = _state.get("risk_history", [])
    hist.append(risk)
    if len(hist) > 120:
        hist = hist[-120:]
    _state["risk_history"] = hist

    return {"near_miss_pairs": near_miss_pairs, "risk_score": risk}


# ── Trajectory predictor ──────────────────────────────────────────────────────
def draw_trajectory_predictions(frame, tracks, horizon=18):
    """
    Dashed linear extrapolation line for each vehicle.
    When two projected paths cross, draw a red X at the intersection.
    """
    active = [t for t in tracks if t.disappeared == 0 and t.speed > 1.5]

    projections = []
    for t in active:
        cx, cy = t.current_centroid()
        px = cx + int(t.vx * horizon)
        py = cy + int(t.vy * horizon)
        projections.append((t, cx, cy, px, py))

    # Arrow drawing removed — direction shown on each track individually

    for i in range(len(projections)):
        for j in range(i+1, len(projections)):
            _, ax1, ay1, ax2, ay2 = projections[i]
            _, bx1, by1, bx2, by2 = projections[j]
            ix, iy = _seg_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
            if ix is not None:
                sz  = 10
                col = _pulse_color(C_RED, 3.0)
                cv2.line(frame, (ix-sz, iy-sz), (ix+sz, iy+sz), col, 2, cv2.LINE_AA)
                cv2.line(frame, (ix+sz, iy-sz), (ix-sz, iy+sz), col, 2, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), sz+6, col, 1, cv2.LINE_AA)


def _draw_dashed_line(frame, p1, p2, color, gap=8, seg_len=6, thickness=2):
    x1, y1  = p1
    x2, y2  = p2
    dx, dy  = x2-x1, y2-y1
    length  = math.hypot(dx, dy)
    if length < 1:
        return
    nx, ny  = dx/length, dy/length
    pos     = 0
    drawing = True
    while pos < length:
        end = min(pos + (seg_len if drawing else gap), length)
        if drawing:
            sx = int(x1 + nx*pos)
            sy = int(y1 + ny*pos)
            ex = int(x1 + nx*end)
            ey = int(y1 + ny*end)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos     = end
        drawing = not drawing


def _seg_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """Return (ix, iy) if the two line segments intersect, else (None, None)."""
    dax, day = ax2-ax1, ay2-ay1
    dbx, dby = bx2-bx1, by2-by1
    denom    = dax*dby - day*dbx
    if abs(denom) < 1e-6:
        return None, None
    dx = bx1-ax1
    dy = by1-ay1
    t  = (dx*dby - dy*dbx) / denom
    u  = (dx*day - dy*dax) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return int(ax1 + t*dax), int(ay1 + t*day)
    return None, None


# ── Near-miss highlights ──────────────────────────────────────────────────────
def draw_near_miss_highlights(frame, near_miss_pairs):
    """Pulsing amber warning badge between near-miss vehicle pairs."""
    for (tid_a, tid_b, dist, mx, my) in near_miss_pairs:
        col  = _pulse_color(C_AMBER, 3.0)
        badge = "! NEAR MISS"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        bx = max(4, mx - bw//2 - 4)
        by = max(bh+8, my - 10)
        ov = frame.copy()
        cv2.rectangle(ov, (bx-2, by-bh-2), (bx+bw+4, by+4), (0, 55, 75), -1)
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (bx-2, by-bh-2), (bx+bw+4, by+4), col, 1)
        cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)


# ── Zone dwell overlays ───────────────────────────────────────────────────────
def draw_zone_dwell_overlays(frame, zones_list, zone_dwells):
    """Fill bars showing loiter progress inside each zone polygon."""
    for zone in zones_list:
        if not hasattr(zone, "loiter_sec") or zone.loiter_sec <= 0:
            continue
        dwells = zone_dwells.get(zone.name, [])
        if not dwells:
            continue
        try:
            pts = np.array(zone.polygon)
            cx  = int(pts[:, 0].mean())
            cy  = int(pts[:, 1].mean())
        except Exception:
            continue

        bar_w, bar_h, gap = 8, 20, 4
        total_w = len(dwells[:4]) * (bar_w + gap) - gap
        bx_start = cx - total_w // 2

        for i, (tid, pct) in enumerate(dwells[:4]):
            bx     = bx_start + i * (bar_w + gap)
            by_bot = cy + 14
            by_top = by_bot - bar_h
            filled = int(bar_h * pct)
            col    = C_RED if pct > 0.8 else (C_AMBER if pct > 0.5 else C_GREEN)
            ov = frame.copy()
            cv2.rectangle(ov, (bx, by_top), (bx+bar_w, by_bot), C_PANEL_BG, -1)
            cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (bx, by_top), (bx+bar_w, by_bot), C_DIVIDER, 1)
            if filled > 0:
                cv2.rectangle(frame, (bx, by_bot-filled), (bx+bar_w, by_bot), col, -1)
            cv2.putText(frame, f"{int(tid):02d}", (bx-1, by_top-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, col, 1, cv2.LINE_AA)


# ── Collision FX ──────────────────────────────────────────────────────────────
def trigger_collision_fx(event, cx: int, cy: int):
    """Set collision FX state. Call once when a collision event is confirmed."""
    _collision_fx["active"] = True
    _collision_fx["cx"]     = cx
    _collision_fx["cy"]     = cy
    _collision_fx["t"]      = 0
    _pulse_rings.append({"cx": cx, "cy": cy, "t": 0, "max_t": 60, "color": C_RED})


def draw_collision_effects(frame):
    """
    Draw all animated collision effects in place.
    Must be called AFTER trigger_collision_fx in the same frame.
    """
    if not _collision_fx["active"]:
        return frame

    t = _collision_fx["t"]

    # White flash only — frames 0-4
    if t < 5:
        alpha = max(0.0, 0.90 - t * 0.18)
        if alpha > 0.02:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
            cv2.addWeighted(ov, alpha, frame, 1.0-alpha, 0, frame)

    _collision_fx["t"] += 1
    if _collision_fx["t"] > 5:
        _collision_fx["active"] = False

    return frame


def _draw_shockwave(frame, cx, cy, t):
    for ring in range(3):
        rt = t - ring * 7
        if rt < 0:
            continue
        radius = int(rt * 5.5)
        if radius > 220:
            continue
        fade = max(0.0, 1.0 - rt / 44.0)
        if rt < 5:
            col = (int(255*fade), int(255*fade), int(255*fade))
        elif rt < 14:
            col = (int(80*fade), int(160*fade), int(255*fade))
        else:
            col = (int(26*fade), int(80*fade), int(255*fade))
        thick = max(1, int(3 * fade))
        if any(c > 4 for c in col):
            cv2.circle(frame, (cx, cy), radius, col, thick, cv2.LINE_AA)



def _draw_pulse_rings(frame):
    to_remove = []
    for ring in _pulse_rings:
        t      = ring["t"]
        max_t  = ring["max_t"]
        if t >= max_t:
            to_remove.append(ring)
            continue
        p       = t / max_t
        radius  = int(p * 92)
        fade    = max(0.0, 1.0 - p * 1.5)
        thick   = max(1, int(3 * (1.0 - p)))
        col     = tuple(int(c * fade) for c in ring["color"])
        if radius > 0 and any(c > 5 for c in col):
            cv2.circle(frame, (ring["cx"], ring["cy"]), radius, col, thick, cv2.LINE_AA)
        ring["t"] += 1
    for r in to_remove:
        _pulse_rings.remove(r)


def _check_accel_pulse(track):
    """Spawn an amber pulse ring when a vehicle accelerates sharply."""
    prev = _track_prev_speed.get(track.id, track.speed)
    _track_prev_speed[track.id] = track.speed
    if track.speed > 6.0 and track.speed > prev * 1.75:
        cx, cy = track.current_centroid()
        _pulse_rings.append({"cx": cx, "cy": cy, "t": 0, "max_t": 30, "color": C_AMBER})


# ── Draw track ────────────────────────────────────────────────────────────────
def draw_track(frame, track, accident=False, seg_mode=True,
               behavior="NORMAL", risk_score=0.0, frame_no=0):
    """Draw one tracked object with all v5 features (1,2,3,11)."""
    h_f, w_f = frame.shape[:2]
    is_ped   = (track.label == "person")
    x1, y1, x2, y2 = track.current_bbox()
    cx, cy          = track.current_centroid()
    conf            = getattr(track, "confidence", 0.80)

    # Role-based color: white/grey=pedestrian, red=moving vehicle, black=stationary, crimson=accident
    color = C_VIOLET if is_ped else get_track_color(track.id, behavior, accident, track.speed)
    # Border thickness: accident stays bold for emphasis; everything else is a
    # slimmer outline so ordinary tracked vehicles don't dominate the frame.
    border_thick = 2 if accident else 1

    # Feature 1 — depth scale (y-position proxy)
    depth_s = _depth_scale(cy, h_f)
    fade_a  = 0.40 + 0.60 * depth_s   # opacity: 0.40 far … 1.0 near

    # ── Segmentation mask ─────────────────────────────────────────────────────
    mask = track.current_mask() if seg_mode else None
    if mask is not None and len(mask) > 2:
        pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
        ov  = frame.copy()
        if accident:
            fill = (30, 0, 20)
        elif behavior == "AGGRESSIVE":
            fill = (0, 35, 80)
        else:
            fill = (20, 28, 38)
        cv2.fillPoly(ov, [pts], fill)
        blend = (0.30 if accident else 0.14) * fade_a
        cv2.addWeighted(ov, blend, frame, 1.0 - blend, 0, frame)
        cv2.polylines(frame, [pts], True, color, border_thick, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thick, cv2.LINE_AA)

    # ── Speed streak ─────────────────────────────────────────────────────────
    if track.speed > 2.5 and len(track.centroids) >= 2:
        pts_list = list(track.centroids)
        n_tail   = min(len(pts_list), 12)
        for k in range(1, n_tail):
            p1    = pts_list[-(k+1)]
            p2    = pts_list[-k]
            frac  = (1.0 - k / n_tail) * fade_a
            dim   = tuple(int(c * frac * 0.85) for c in color)
            cv2.line(frame, p1, p2, dim, max(1, int(2*frac)), cv2.LINE_AA)

    # ── Inner ring — spec: scaleY(0.62), border = 65% of track color ───────────
    rx = max(20, (x2 - x1) // 2 + 8)
    ry = max(8,  int(rx * 0.62))       # HTML spec: scaleY(0.62) constant
    ring_thick = 2 if accident else 1
    # color-mix(tcolor 65%, transparent) at fade_a
    ring_col = tuple(int(c * 0.65 * fade_a) for c in color)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, ring_col, ring_thick, cv2.LINE_AA)


    # Gyro rings on accident/critical tracks (extra flat ellipses, one reversed)
    if accident:
        gr = rx + 20
        gry = int(gr * 0.32)   # very flat: scaleY(0.32)
        gyro_col = tuple(int(c * 0.30 * fade_a) for c in color)
        gyro_ang = int(time.time() * 40) % 360
        cv2.ellipse(frame, (cx, cy), (gr, gry), gyro_ang, 0, 360, gyro_col, 1, cv2.LINE_AA)
        gyro_col2 = tuple(int(c * 0.18 * fade_a) for c in color)
        cv2.ellipse(frame, (cx, cy), (gr, gry), -gyro_ang, 0, 360, gyro_col2, 1, cv2.LINE_AA)

    # Centroid dot
    cv2.circle(frame, (cx, cy), 2, ring_col, -1, cv2.LINE_AA)

    # ── Direction arrow — small filled triangle just outside the ellipse ring ─
    # (previously a long arrowedLine from the centroid, which looked messy
    # once several tracks' shafts crossed each other — now a compact solid
    # chevron pinned to the ring edge, same idea as a heading indicator)
    cents = list(track.centroids)
    if len(cents) >= 5 and track.speed > 1.5:
        dx = cents[-1][0] - cents[-5][0]
        dy = cents[-1][1] - cents[-5][1]
        mag = math.hypot(dx, dy)
        if mag > 2:
            ux, uy   = dx / mag, dy / mag
            perp_x, perp_y = -uy, ux
            tip_dist = rx + 12    # tip sits just clear of the ellipse ring
            tri_len, tri_half = 8, 4
            tip   = (cx + ux*tip_dist,               cy + uy*tip_dist)
            base  = (tip[0] - ux*tri_len,             tip[1] - uy*tri_len)
            left  = (base[0] + perp_x*tri_half,       base[1] + perp_y*tri_half)
            right = (base[0] - perp_x*tri_half,       base[1] - perp_y*tri_half)
            arr_col = tuple(int(c * fade_a) for c in color)
            tri = np.array([tip, left, right], dtype=np.int32)
            cv2.fillConvexPoly(frame, tri, arr_col, cv2.LINE_AA)


    # ── Label tag — format "04 · 14km/h" with middle dot via PIL ─────────────
    fps_val   = _state.get("video_fps", 25.0)
    ppm       = _state.get("pixels_per_meter", 8.0)
    speed_kmh = track.speed / max(0.1, ppm) * fps_val * 3.6
    label_txt = f"{int(track.id):02d} \xb7 {speed_kmh:.0f}km/h"  # · = U+00B7

    # Tag position: below outer ring
    tag_y = cy + ry + 14
    tag_x = cx - 30

    # Tag background (rgba(3,10,15,0.85) with 55% color border)
    # Stationary vehicles use a near-black box border, but the label itself
    # must stay legible against the dark tag background — fall back to a
    # neutral grey instead of rendering black-on-black.
    tag_col = (_pulse_color(C_CRIMSON, 4.0) if accident else
               C_MIST if color == C_BLACK_BORDER else color)
    border_col = tuple(int(c * 0.55 * fade_a) for c in tag_col)
    # Estimate tag width for background (PIL will render precise text)
    approx_tw = len(label_txt) * 7
    cv2.rectangle(frame, (tag_x - 3, tag_y - 13), (tag_x + approx_tw + 4, tag_y + 3),
                  (15, 10, 8), -1)
    cv2.rectangle(frame, (tag_x - 3, tag_y - 13), (tag_x + approx_tw + 4, tag_y + 3),
                  border_col, 1)

    txt_col = _pulse_color(C_CRIMSON, 4.0) if accident else tag_col
    _pil_label(frame, label_txt, tag_x, tag_y, txt_col, size="sm")

    # behavior tag (top of bbox)
    btag = BEHAVIOR_TAG.get(behavior)
    if btag and not is_ped:
        btag_col = (_pulse_color(C_CRIMSON, 5.0) if behavior == "ERRATIC"
                    else _pulse_color(C_AMBER, 2.5) if behavior == "AGGRESSIVE"
                    else C_MIST)
        cv2.putText(frame, btag, (cx - 14, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, btag_col, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# FEED OVERLAY FUNCTIONS  (features 8, 9, 10, 12 — drawn on the video region)
# ══════════════════════════════════════════════════════════════════════════════

def draw_feed_grid(frame):
    """64px cyan grid at ~5% opacity — matches HTML .feed-grid background-size:64px."""
    h, w = frame.shape[:2]
    step = 64
    ov   = frame.copy()
    # C_LINE_FULL = BGR(255,220,90) = CSS rgba(90,220,255) = light cyan
    for x in range(0, w, step):
        cv2.line(ov, (x, 0), (x, h), C_LINE_FULL, 1)
    for y in range(0, h, step):
        cv2.line(ov, (0, y), (w, y), C_LINE_FULL, 1)
    cv2.addWeighted(ov, 0.05, frame, 0.95, 0, frame)


def draw_feed_scanline(frame, frame_no):
    """Moving horizontal glow strip — CSS scanline effect."""
    h, w = frame.shape[:2]
    y = int((frame_no * 4) % (h + 100)) - 100
    band_h = 80
    y1 = max(0, y)
    y2 = min(h, y + band_h)
    if y2 <= y1:
        return
    ov = frame[y1:y2].copy()
    ov = np.clip(ov.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    cv2.addWeighted(ov, 0.08, frame[y1:y2], 0.92, 0, frame[y1:y2])


def draw_feed_vignette(frame):
    """Radial edge darkening — precomputed and cached."""
    h, w = frame.shape[:2]
    key  = (h, w)
    if key not in _vignette_cache:
        yv = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xv = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        GX, GY  = np.meshgrid(xv, yv)
        mask    = np.clip((GX**2 + GY**2) / (0.55**2), 0, 1) * 0.65
        _vignette_cache[key] = mask[:, :, np.newaxis]
    alpha = _vignette_cache[key]
    frame[:] = np.clip(frame.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def draw_radar(frame, tracks, frame_h, frame_w):
    """Feature 8 — mini bird's-eye radar, bottom-right of video."""
    R   = 52   # radius px
    cx  = frame_w - R - 16
    cy  = frame_h - R - 52   # above filmstrip

    # Background
    cv2.circle(frame, (cx, cy), R, (8, 6, 4), -1)
    cv2.circle(frame, (cx, cy), R, tuple(int(c*0.35) for c in C_CYAN), 1)
    for r in (R*1//3, R*2//3):
        cv2.circle(frame, (cx, cy), r, tuple(int(c*0.18) for c in C_CYAN), 1)

    # Sweep line
    ang = (time.time() * 55) % 360
    sx  = int(cx + R * math.cos(math.radians(ang)))
    sy  = int(cy + R * math.sin(math.radians(ang)))
    cv2.line(frame, (cx, cy), (sx, sy), C_CYAN, 1, cv2.LINE_AA)

    # Self dot
    cv2.circle(frame, (cx, cy), 3, C_TEXT, -1)

    # Vehicle blips
    for t in tracks:
        if t.disappeared > 0:
            continue
        tx, ty = t.current_centroid()
        bx = cx + int((tx / max(1, frame_w) - 0.5) * R * 1.8)
        by = cy + int((ty / max(1, frame_h) - 0.5) * R * 1.8)
        if math.hypot(bx-cx, by-cy) <= R:
            bc = _track_color_map.get(t.id, C_CYAN)
            cv2.circle(frame, (bx, by), 3, bc, -1)
            cv2.circle(frame, (bx, by), 5, tuple(int(c*0.4) for c in bc), 1)

    cv2.circle(frame, (cx, cy), R, tuple(int(c*0.35) for c in C_CYAN), 1)
    cv2.putText(frame, "BIRD'S-EYE", (cx - R, cy - R - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_MIST, 1, cv2.LINE_AA)


def draw_filmstrip(frame, frame_no, accident_frames_list):
    """Feature 9 — rolling frame thumbnail strip at bottom of video."""
    h, w   = frame.shape[:2]
    strip_h = 34
    y0      = h - strip_h
    n       = 14
    tw      = (w - 8) // n

    # Background
    ov = frame[y0:h].copy()
    ov[:] = (6, 4, 3)
    cv2.addWeighted(ov, 0.72, frame[y0:h], 0.28, 0, frame[y0:h])
    cv2.line(frame, (0, y0), (w, y0), tuple(int(c*0.28) for c in C_CYAN), 1)

    acc_set = set(accident_frames_list[-20:])
    for i in range(n):
        fn  = frame_no - (n - 1 - i)
        if fn < 0:
            continue
        tx  = 4 + i * tw
        is_now   = (fn == frame_no)
        is_spike = fn in acc_set

        if is_now:
            bc, fc = (0, 30, 40), C_CYAN
        elif is_spike:
            bc, fc = (8, 2, 12), C_CRIMSON
        else:
            bc, fc = (10, 8, 6), tuple(int(c*0.22) for c in C_CYAN)

        cv2.rectangle(frame, (tx, y0+2), (tx+tw-2, h-2), bc, -1)
        cv2.rectangle(frame, (tx, y0+2), (tx+tw-2, h-2), fc, 1)
        if is_spike:
            mid = (tx + tw//2, y0 + strip_h//2)
            cv2.line(frame, (mid[0], y0+4), (mid[0], h-4), C_CRIMSON, 1)
        lbl = str(fn % 1000)
        cv2.putText(frame, lbl, (tx+2, h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, (130, 120, 110), 1)


def draw_accident_banner(frame, event):
    """Feature 12 — flashing critical banner overlaid on video, bottom-left."""
    if event is None:
        return
    h, w = frame.shape[:2]
    ids_str = " + ".join(str(i) for i in getattr(event, "vehicle_ids", []))
    fn_str  = f"FRAME {_state['frame_no']}"
    label   = "ACCIDENT DETECTED"

    font = cv2.FONT_HERSHEY_DUPLEX
    (lw, lh), _ = cv2.getTextSize(label, font, 0.52, 1)
    bw, bh = lw + 60, 52
    bx, by = 16, max(4, h - 110)   # clamp so a short/cropped frame can't push this negative

    ov = frame.copy()
    cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), (10, 2, 25), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

    pulse  = _pulse(1.1, 0.45, 1.0)
    border = tuple(int(c * pulse) for c in C_CRIMSON)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), border, 1)

    # Pulsing dot
    dot_col = _pulse_color(C_CRIMSON, 1.25)
    cv2.circle(frame, (bx+14, by+bh//2), 5, dot_col, -1)

    # Text
    txt_col = _pulse_color(C_CRIMSON, 4.0)
    cv2.putText(frame, label, (bx+26, by+22), font, 0.52, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (bx+26, by+22), font, 0.52, txt_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TRACKS {ids_str}  ·  {fn_str}", (bx+26, by+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_MIST, 1, cv2.LINE_AA)


# ── Side panel helpers ────────────────────────────────────────────────────────
def _panel_rect(canvas, x1, y1, x2, y2, fill=C_PANEL_BG, alpha=0.92, border=True):
    ov = canvas.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(ov, alpha, canvas, 1-alpha, 0, canvas)
    if border:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_PANEL_BORDER, 1)

def _label(canvas, text, x, y, color=C_TEXT_PRIMARY, scale=0.42, thick=1):
    cv2.putText(canvas, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def _divider(canvas, y, x1=None, x2=None):
    x1 = x1 if x1 is not None else PANEL_X + 10
    x2 = x2 if x2 is not None else CANVAS_W - 10
    cv2.line(canvas, (x1, y), (x2, y), C_DIVIDER, 1)


def _pill_bar(canvas, x, y, total_w, bar_h, fill_w, bg_col, fg_col):
    """Horizontal bar with rounded pill ends. fill_w = filled pixels."""
    r = bar_h // 2
    tw = max(r * 2 + 1, total_w)
    cv2.rectangle(canvas, (x + r, y), (x + tw - r, y + bar_h), bg_col, -1)
    cv2.circle(canvas, (x + r, y + r), r, bg_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + tw - r, y + r), r, bg_col, -1, cv2.LINE_AA)
    fw = max(0, min(fill_w, tw))
    if fw <= 0:
        return
    if fw >= tw - r:
        cv2.rectangle(canvas, (x + r, y), (x + tw - r, y + bar_h), fg_col, -1)
        cv2.circle(canvas, (x + r, y + r), r, fg_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x + tw - r, y + r), r, fg_col, -1, cv2.LINE_AA)
    elif fw > r:
        cv2.rectangle(canvas, (x + r, y), (x + fw, y + bar_h), fg_col, -1)
        cv2.circle(canvas, (x + r, y + r), r, fg_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x + fw, y + r), r, fg_col, -1, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (x + r, y + r), r, fg_col, -1, cv2.LINE_AA)


# ── Side panel ────────────────────────────────────────────────────────────────
def draw_side_panel(canvas):  # noqa: C901
    px = PANEL_X          # 1216
    L  = px + 18          # left content (18px padding matches HTML)
    R  = CANVAS_W - 18    # right content
    CW = R - L            # 284px usable width

    # Panel background: rgba(11,20,30) ≈ BGR(30,20,11) over void
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), C_PANEL_BG, -1)
    cv2.line(canvas, (px, 0), (px, CANVAS_H), C_LINE_FULL, 1)

    y = TOP_BAR_H

    # ── local helpers ─────────────────────────────────────────────────────────
    def sep():
        nonlocal y
        cv2.line(canvas, (px, y), (CANVAS_W, y), C_LINE, 1)
        y += 1

    def hdr(title):
        nonlocal y
        y += 8
        # Glow behind tick (CSS: box-shadow: 0 0 6px var(--cyan))
        glow_c = tuple(int(c * 0.22) for c in C_CYAN)
        cv2.rectangle(canvas, (L-2, y-2), (L+5, y+15), glow_c, -1)
        # Bright 3x13px accent tick
        cv2.rectangle(canvas, (L, y), (L+3, y+13), C_CYAN, -1)
        # Title: Orbitron 13px, red (was mist/grey, and 11px)
        _pil_label(canvas, title, L+12, y+13, C_CYAN, size=13, face="orb", weight=700)
        y += 21

    def bot():
        nonlocal y
        y += 10  # CARD_PAD_BOT — generous gap so cards stay visually separated
                 # even when the fullscreen window upscales/blurs small text
        sep()

    def rval_cv(text, yy, col, sc=0.38):
        (tw2, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
        cv2.putText(canvas, text, (R-tw2, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, 1, cv2.LINE_AA)

    def rval_pil(text, yy, col, size="xs", face="jb"):
        tw2, _ = _pil_measure(text, size=size, face=face)
        if tw2 == 0:
            rval_cv(text, yy, col)
            return
        _pil_label(canvas, text, R-tw2, yy, col, size=size, face=face)

    def stat_row(key, val, vc=None):
        """HTML: .stat-key = JetBrains Mono 10.5px mist; .stat-val = 12px text weight 500."""
        nonlocal y
        _pil_label(canvas, key, L, y, C_MIST, size=12, face="jb", weight=400)
        vw, _ = _pil_measure(val, size=13, face="jb", weight=500)
        _pil_label(canvas, val, R-vw, y, vc if vc else C_TEXT, size=13, face="jb", weight=500)
        y += 15  # row height for the larger 12/13px text

    # ═══════════════════════════════════════════════════════════════════════════
    # CARD 1 — SYSTEM  (trimmed: no raw frame counter — meaningless to a viewer
    # who isn't reading source code; uptime/mode/TensorRT/fps tell the "is this
    # a real, live, GPU-accelerated system" story on their own)
    # ═══════════════════════════════════════════════════════════════════════════
    hdr("SYSTEM")
    uptime  = int(time.time() - _state["session_start"])
    trt_on  = _state.get("use_tensorrt", False)
    fps     = _state["fps"]
    # Body text is white throughout (color reserved for headers/dividers/
    # alerts/wordmark) — status is conveyed by the badges' fill below, not
    # by tinting these values.
    stat_row("UPTIME",   f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}")
    stat_row("MODE",     _state["mode"])
    stat_row("TENSORRT", "ON" if trt_on else "OFF")
    stat_row("FPS",      f"{fps:.1f}")

    # FPS bar — HTML: height:5px; background:linear-gradient(90deg,--crimson,--amber)
    # margin-top:8px from .stat-row baseline to bar; we use 5px (compact)
    y += 5
    BW  = CW
    BAR = 5
    ff  = int(BW * min(fps, 30) / 30)
    # Track background: dark void-ish (#0d0c08 ≈ BGR 8,12,13)
    cv2.rectangle(canvas, (L, y), (L+BW, y+BAR), (8, 12, 13), -1)
    # Gradient fill: crimson BGR(92,59,255) → amber BGR(32,176,255)
    if ff > 1:
        for xi in range(ff):
            t = xi / max(ff - 1, 1)
            col_i = (int(92 + (32-92)*t), int(59 + (176-59)*t), 255)
            cv2.line(canvas, (L+xi, y), (L+xi, y+BAR-1), col_i, 1)
    y += BAR

    # Hex badges — 2 now (MODE, TRT); space-between across the card width
    y += 8
    HW, HH = 74, 34
    for bi, (bval, blbl) in enumerate([
        (_state["mode"],            "MODE"),
        ("ON" if trt_on else "OFF", "TRT"),
    ]):
        bx = L if bi == 0 else R - HW
        by = y
        pts_h = np.array([
            [bx + HW//2, by],
            [bx + HW,    by + HH//4],
            [bx + HW,    by + 3*HH//4],
            [bx + HW//2, by + HH],
            [bx,         by + 3*HH//4],
            [bx,         by + HH//4],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_h], (48, 37, 14))
        cv2.polylines(canvas, [pts_h], True, (80, 60, 20), 1, cv2.LINE_AA)
        cx = bx + HW // 2
        lw, _ = _pil_measure(blbl, size=10, face="orb", weight=500)
        _pil_label(canvas, blbl, cx - lw//2, by + 14,       C_MIST, size=10, face="orb", weight=500)
        vw, _ = _pil_measure(bval, size=13, face="orb", weight=700)
        _pil_label(canvas, bval, cx - vw//2, by + HH - 4, C_TEXT, size=13, face="orb", weight=700)
    y += HH
    bot()

    # ═══════════════════════════════════════════════════════════════════════════
    # CARD 3 — ACTIVE DETECTIONS
    # HTML: .det-grid { grid-template-columns:1fr 1fr; gap:8px; margin-top:8px }
    #       .det-num  { Orbitron 22px weight 700 }   .det-lbl { JB Mono 9px mist }
    #       .near-miss-box { margin-top:8px; bg:rgba(255,176,32,0.06) }
    # Colors pre-blended over panel BGR(30,20,11):
    #   cyan-bg   rgba(47,230,255,0.06)  → BGR(44,33,13)
    #   cyan-bdr  rgba(90,220,255,0.12)  → BGR(57,44,20)
    #   amber-bg  rgba(255,176,32,0.06)  → BGR(30,29,26)
    #   amber-bdr rgba(255,176,32,0.15)  → BGR(30,43,48)
    #   amber-bg-active  .12             → BGR(30,39,40)
    #   amber-bdr-active .35             → BGR(31,75,96)
    # ═══════════════════════════════════════════════════════════════════════════
    hdr("ACTIVE DETECTIONS")
    nm_total = _state.get("near_miss_total", 0)
    hw = (CW - 8) // 2   # half-width, gap=8px between two columns
    BH = 34               # box height (grown from 26 for larger num/label text)

    y += 4   # det-grid margin-top (HTML: 8px, compact to 4px)

    def _det_box(bx, by, bw, bh, num, lbl, nc, bg, bdr):
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), bg,  -1)
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), bdr,  1)
        nw, nh = _pil_measure(num, size=18, face="orb", weight=700)
        _pil_label(canvas, num, bx+(bw-nw)//2, by+nh+4, nc, size=18, face="orb", weight=700)
        lw, _  = _pil_measure(lbl, size=11, face="jb", weight=400)
        _pil_label(canvas, lbl, bx+(bw-lw)//2, by+bh-3, C_MIST, size=11, face="jb", weight=400)

    # VEHICLES / PEDESTRIANS — numbers are body text (white); the box tint
    # still carries the cyan/red chrome, just not the digits themselves.
    _det_box(L,      y, hw, BH,
             str(_state["n_vehicles"]), "VEHICLES", C_TEXT,
             (44,33,13), (57,44,20))
    _det_box(L+hw+8, y, hw, BH,
             str(_state["n_pedestrians"]), "PEDESTRIANS", C_TEXT,
             (44,33,13), (57,44,20))
    y += BH + 4   # gap to near-miss box (HTML: margin-top:8px, compact to 4px)

    # NEAR-MISS EVENTS — box tint still goes amber when active (a real,
    # at-a-glance status cue); the count itself stays white body text.
    if nm_total > 0:
        nm_bg  = (30, 39, 40)   # rgba(255,176,32,0.12) active
        nm_bdr = (31, 75, 96)   # rgba(255,176,32,0.35) active
    else:
        nm_bg  = (30, 29, 26)   # rgba(255,176,32,0.06) normal
        nm_bdr = (30, 43, 48)   # rgba(255,176,32,0.15) normal
    nm_nc = C_TEXT
    _det_box(L, y, CW, BH,
             str(nm_total), "NEAR-MISS EVENTS", nm_nc,
             nm_bg, nm_bdr)
    y += BH
    bot()

    # ═══════════════════════════════════════════════════════════════════════════
    # CARD 3 — THREAT LEVEL — hero speedometer.
    # Deliberately the visual anchor of the panel: a car-speedometer-style
    # gauge (eased needle + tick marks + glowing orange->red arc, no green/
    # amber) reading collision risk 0-100%, with the risk trend underneath
    # and a "time to impact" callout that only appears with a real
    # converging pair (with hysteresis so a one-frame gap in the data
    # doesn't flicker it — see below).
    # ═══════════════════════════════════════════════════════════════════════════
    hdr("THREAT LEVEL")
    hist     = _state.get("risk_history", [])
    cur_risk = hist[-1] if hist else 0.0
    threat   = ("CRITICAL" if cur_risk > 0.7 else
                "ELEVATED" if cur_risk > 0.4 else "NOMINAL")

    y += 12
    GR      = 92                        # gauge radius
    GCX     = L + CW // 2               # centered horizontally in the card
    GCY     = y + GR
    START_A = 135.0                     # cv2 angle convention (clockwise from east)
    SWEEP   = 270.0                     # classic automotive gauge sweep, gap at bottom
    ARC_W   = 7                         # slim arc (was 11)

    # Needle eases toward the target angle instead of snapping — smooth
    # tachometer-style sweep. Persisted in _state so it carries across frames.
    target_a = START_A + max(0.0, min(1.0, cur_risk)) * SWEEP
    cur_a    = _state.get("_gauge_angle")
    cur_a    = target_a if cur_a is None else cur_a + (target_a - cur_a) * 0.22
    _state["_gauge_angle"] = cur_a
    needle_frac = (cur_a - START_A) / SWEEP
    needle_col  = _gauge_color(needle_frac)

    # Background track (full sweep, dim neutral — only the filled portion
    # carries the orange->red color, like an unlit vs lit tachometer face)
    cv2.ellipse(canvas, (GCX, GCY), (GR, GR), 0, START_A, START_A + SWEEP,
                (22, 18, 15), ARC_W, cv2.LINE_AA)

    # Soft glow behind the animated progress arc (blurred, additive) —
    # gives the "slim glowing arc" a futuristic HUD feel instead of a flat
    # line. Scoped to a small ROI around the gauge for performance.
    if cur_a > START_A + 1:
        pad = 16
        bx1, by1 = max(0, GCX-GR-pad), max(0, GCY-GR-pad)
        bx2, by2 = min(CANVAS_W, GCX+GR+pad), min(CANVAS_H, GCY+GR+pad)
        if bx2 > bx1 and by2 > by1:
            glow = np.zeros((by2-by1, bx2-bx1, 3), dtype=np.uint8)
            gcx_l, gcy_l = GCX-bx1, GCY-by1
            steps = max(2, int((cur_a - START_A) / 4))
            for si in range(steps):
                a0 = START_A + (cur_a-START_A) * si/steps
                a1 = START_A + (cur_a-START_A) * (si+1)/steps
                seg_col = _gauge_color(((a0+a1)/2 - START_A) / SWEEP)
                cv2.ellipse(glow, (gcx_l, gcy_l), (GR, GR), 0, a0, a1, seg_col, 6, cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (13, 13), 0)
            roi = canvas[by1:by2, bx1:bx2].astype(np.float32)
            canvas[by1:by2, bx1:bx2] = np.clip(roi + glow.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)

        # Crisp gradient arc on top of the glow — orange (low) -> red (high),
        # drawn in small segments since cv2.ellipse only takes one color.
        steps = max(2, int((cur_a - START_A) / 4))
        for si in range(steps):
            a0 = START_A + (cur_a-START_A) * si/steps
            a1 = START_A + (cur_a-START_A) * (si+1)/steps
            seg_col = _gauge_color(((a0+a1)/2 - START_A) / SWEEP)
            cv2.ellipse(canvas, (GCX, GCY), (GR, GR), 0, a0, a1, seg_col, ARC_W, cv2.LINE_AA)

    # Tick marks every 10%
    for i in range(11):
        ang    = math.radians(START_A + (i / 10.0) * SWEEP)
        r1, r2 = GR - 16, GR - 4
        x1, y1 = int(GCX + r1 * math.cos(ang)), int(GCY + r1 * math.sin(ang))
        x2, y2 = int(GCX + r2 * math.cos(ang)), int(GCY + r2 * math.sin(ang))
        cv2.line(canvas, (x1, y1), (x2, y2), C_MIST, 1, cv2.LINE_AA)

    # Tapered needle (triangle, not a uniform line) — shortened so its tip
    # can never reach the readout zone below (see clearance note at the
    # number placement), regardless of angle.
    n_ang    = math.radians(cur_a)
    tip_len  = GR - 34
    hub_w    = 5
    perp     = n_ang + math.pi / 2
    tip      = (GCX + tip_len*math.cos(n_ang),      GCY + tip_len*math.sin(n_ang))
    base1    = (GCX + hub_w*math.cos(perp),          GCY + hub_w*math.sin(perp))
    base2    = (GCX - hub_w*math.cos(perp),          GCY - hub_w*math.sin(perp))
    needle_pts = np.array([tip, base1, base2], dtype=np.int32)
    cv2.fillPoly(canvas, [needle_pts], needle_col)
    cv2.polylines(canvas, [needle_pts], True, needle_col, 1, cv2.LINE_AA)
    cv2.circle(canvas, (GCX, GCY), 7, (16, 13, 11), -1, cv2.LINE_AA)
    cv2.circle(canvas, (GCX, GCY), 7, needle_col, 2, cv2.LINE_AA)

    # Big bold % readout — always red (per design direction, not risk-
    # colored) and status word tinted along the same orange->red scale as
    # the arc. Clearance note: the needle's deepest possible reach into the
    # gauge's bottom gap is tip_len*sin(135°) ≈ tip_len*0.707 below center,
    # which with tip_len=GR-34 is well above where the readout starts —
    # true at every angle the needle can take, not just the current one.
    pct_s  = f"{int(round(cur_risk*100))}%"
    pw, ph = _pil_measure(pct_s, size=30, face="orb", weight=800)
    if pw == 0: pw, ph = 56, 24
    num_y = GCY + 50 + ph
    _pil_label(canvas, pct_s, GCX - pw//2, num_y, C_GAUGE_HIGH, size=30, face="orb", weight=800)
    tlw, _ = _pil_measure(threat, size=13, face="orb", weight=700)
    _pil_label(canvas, threat, GCX - tlw//2, num_y + 20, _gauge_color(cur_risk), size=13, face="orb", weight=700)

    y = num_y + 30

    # Time-to-impact callout — hysteresis + eased presence so it never hard-
    # flickers: a one-frame gap in ttc_sec (noisy track data) holds the last
    # known reading for GRACE_SEC before the block starts easing its height
    # back to 0, so content below it slides smoothly instead of the block
    # vanishing and reappearing every few frames.
    now = time.time()
    ttc_raw = _state.get("ttc_sec", None)
    if ttc_raw is not None:
        _state["_ttc_last_seen"] = now
        _state["_ttc_last_val"]  = ttc_raw
        _state["_ttc_last_pair"] = _state.get("ttc_pair", None)
    GRACE_SEC   = 1.2
    have_recent = (now - _state.get("_ttc_last_seen", 0.0)) < GRACE_SEC
    target_anim = 1.0 if have_recent else 0.0
    anim        = _state.get("_ttc_anim", 0.0)
    anim       += (target_anim - anim) * 0.30
    _state["_ttc_anim"] = anim

    if anim > 0.02:
        ttc_s = _state.get("_ttc_last_val")
        tp    = _state.get("_ttc_last_pair")
        if ttc_s is not None:
            urgent = ttc_s < 4.0
            base_col = C_CRIMSON if ttc_s < 2.0 else (C_AMBER if urgent else C_TEXT)
            fade_col = tuple(int(c * anim) for c in base_col)
            block_h  = int(24 * anim)
            y += int(8 * anim)
            if urgent:
                alpha = 0.55 * anim
                ov = canvas.copy()
                cv2.rectangle(ov, (L, y - 4), (R, y - 4 + block_h), (18, 6, 10), -1)
                cv2.addWeighted(ov, alpha, canvas, 1 - alpha, 0, canvas)
                cv2.rectangle(canvas, (L, y - 4), (R, y - 4 + block_h),
                              tuple(int(c * 0.5) for c in fade_col), 1)
            msg   = f"IMPACT IN {ttc_s:.1f}s" + (f"   TRK {tp[0]} vs {tp[1]}" if tp else "")
            mw, _ = _pil_measure(msg, size=12, face="jb", weight=700)
            _pil_label(canvas, msg, GCX - mw//2, y - 4 + block_h - 6, fade_col, size=12, face="jb", weight=700)
            y += block_h
    else:
        _state["_ttc_last_val"]  = None
        _state["_ttc_last_pair"] = None

    # Risk trend sparkline — themed to the gauge's orange->red scale instead
    # of green; "RISK TREND" label + value are body text, so white.
    y += 12
    GH = 34
    hist_d = hist[-CW:]
    trend_col = _gauge_color(cur_risk)
    if len(hist_d) >= 2:
        n_d = len(hist_d)
        pts_g = [(L + i*(CW-1)//(n_d-1), y + GH - 2 - int(rv*(GH-4)))
                 for i, rv in enumerate(hist_d)]
        arr_g = np.array(pts_g, dtype=np.int32)
        fmask = np.zeros((GH, CW), dtype=np.uint8)
        local_fill = np.array(
            [(p[0]-L, p[1]-y) for p in pts_g]
            + [(pts_g[-1][0]-L, GH), (pts_g[0][0]-L, GH)], dtype=np.int32)
        cv2.fillPoly(fmask, [local_fill], 255)
        fill_bgr = np.array([float(trend_col[0]), float(trend_col[1]), float(trend_col[2])])
        for ri in range(GH):
            al = 0.30 * (1.0 - ri / max(GH-1, 1))
            if al < 0.01: continue
            fc = np.where(fmask[ri] > 0)[0]
            if fc.size == 0: continue
            x1c, x2c = int(fc[0]), int(fc[-1])
            reg = canvas[y+ri, L+x1c:L+x2c+1].astype(np.float32)
            canvas[y+ri, L+x1c:L+x2c+1] = np.clip(
                reg*(1-al) + fill_bgr*al, 0, 255).astype(np.uint8)
        cv2.polylines(canvas, [arr_g], False, trend_col, 2, cv2.LINE_AA)
    y += GH + 16   # clears the trend line even when it sits low in the graph
                   # (a low risk value pushes the line near the bottom edge)
    lbl_txt = "RISK TREND  "
    risk_pct = f"{int(cur_risk*100)}%"
    rw, _ = _pil_measure(risk_pct, size=11, face="jb", weight=700)
    lw, _ = _pil_measure(lbl_txt,  size=11, face="jb", weight=400)
    _pil_label(canvas, lbl_txt,  L, y, C_MIST, size=11, face="jb", weight=400)
    _pil_label(canvas, risk_pct, L + lw, y, C_TEXT, size=11, face="jb", weight=700)
    y += 11
    bot()

    # ═══════════════════════════════════════════════════════════════════════════
    # CARD 4 — ZONES — rebuilt to actually name the monitored zones and show a
    # live status per zone (previously this card was only a heatmap toggle,
    # which didn't answer "what does this show?" for an outside viewer).
    # ═══════════════════════════════════════════════════════════════════════════
    hdr("ZONES")
    zone_status = _state.get("zone_status", [])
    y += 4
    if not zone_status:
        _pil_label(canvas, "NO ZONES CONFIGURED", L, y + 10, C_MIST, size=12, face="jb", weight=400)
        y += 16
    else:
        for zs in zone_status:
            active     = zs["occupied"] > 0
            # Status color lives in the dot only — the text itself is white
            # body text either way.
            dot_col    = C_AMBER if active else tuple(int(c*0.45) for c in C_GREEN)
            status_txt = f"{zs['occupied']} TRACKED" if active else "CLEAR"
            cv2.circle(canvas, (L + 4, y - 3), 4, dot_col, -1, cv2.LINE_AA)
            _pil_label(canvas, zs["name"].upper(), L + 15, y, C_TEXT, size=12, face="jb", weight=500)
            stw, _ = _pil_measure(status_txt, size=11, face="jb", weight=600)
            _pil_label(canvas, status_txt, R - stw, y, C_TEXT, size=11, face="jb", weight=600)
            y += 19

    # Speed-heatmap toggle — compact secondary row (still a real H-key
    # feature). Text white; the toggle knob itself still carries color.
    y += 5
    hm_on  = _state.get("heatmap_on", False)
    zm_lbl = "SPEED HEATMAP  "
    zs_lbl = "ON" if hm_on else "OFF"
    zlw, _ = _pil_measure(zm_lbl, size=11, face="jb", weight=400)
    _pil_label(canvas, zm_lbl, L, y, C_MIST, size=11, face="jb", weight=400)
    _pil_label(canvas, zs_lbl, L + zlw, y, C_TEXT, size=11, face="jb", weight=600)
    TWp, THp = 30, 15
    TXp, TYp = R - TWp, y - 11
    cv2.rectangle(canvas, (TXp, TYp), (TXp+TWp, TYp+THp), (13,32,41) if hm_on else (12,12,12), -1)
    cv2.rectangle(canvas, (TXp, TYp), (TXp+TWp, TYp+THp), (30,30,30), 1)
    knob_x = TXp + TWp - 7 if hm_on else TXp + 7
    cv2.circle(canvas, (knob_x, TYp + THp//2), 5, C_GREEN if hm_on else C_MIST, -1, cv2.LINE_AA)
    y += 10
    bot()

    # ═══════════════════════════════════════════════════════════════════════════
    # CARD 9 — INCIDENT LOG  (last card — no bottom border)
    # ═══════════════════════════════════════════════════════════════════════════
    hdr("INCIDENT LOG")
    events = list(reversed(_state["event_log"]))
    if not events:
        # Dashed border box — stretched to fill the rest of the sidebar's
        # height instead of a small fixed box that left a dead gap above
        # the footer credit line.
        bx1, by1 = L, y + 4
        bx2, by2 = R, max(y + 28, CANVAS_H - BOT_BAR_H - 18)
        for xi3 in range(bx1, bx2, 8):
            cv2.line(canvas, (xi3, by1), (min(xi3+5, bx2), by1), C_LINE, 1)
            cv2.line(canvas, (xi3, by2), (min(xi3+5, bx2), by2), C_LINE, 1)
        for yi3 in range(by1, by2, 8):
            cv2.line(canvas, (bx1, yi3), (bx1, min(yi3+5, by2)), C_LINE, 1)
            cv2.line(canvas, (bx2, yi3), (bx2, min(yi3+5, by2)), C_LINE, 1)
        em_txt = "NO INCIDENTS RECORDED"
        etw, _ = _pil_measure(em_txt, size=12, face="jb", weight=400)
        _pil_label(canvas, em_txt, L + (CW-etw)//2, (by1+by2)//2 + 4, C_MIST, size=12, face="jb", weight=400)
        y = by2 + 4
    else:
        # Single-line-per-event format, always full brightness (no age fade,
        # no pulsing) so every recent incident stays legible, and a fixed
        # row height so several events reliably fit above the credit line
        # regardless of how much vertical room earlier cards consumed.
        y += 6   # matches the empty-state branch's margin below the header
        max_y = CANVAS_H - BOT_BAR_H - 16   # same floor as the empty-state box
        ROW_H = 16
        for i, ev in enumerate(events[:5]):
            # Always draw at least the most recent incident, even if earlier
            # cards left little room — a cramped-but-visible entry beats a
            # silently empty card.
            if i > 0 and y + ROW_H > max_y: break
            age = time.time() - ev["epoch"]
            sc3 = {"HIGH":C_CRIMSON,"MEDIUM":C_AMBER,"LOW":C_MIST}.get(ev["sev"], C_MIST)
            if age < 10:
                ov3 = canvas.copy()
                cv2.rectangle(ov3, (PANEL_X+4, y-4), (CANVAS_W-4, y+ROW_H-5), (20,4,8), -1)
                cv2.addWeighted(ov3, 0.28, canvas, 0.72, 0, canvas)
            sev3 = f"[{ev['sev'][:3]}]"
            sw3, _ = _pil_measure(sev3, size=11, face="jb", weight=400)
            left_txt = f"{ev['ts']}  IDs {ev['ids']}  {ev['flags']}"
            avail = CW - sw3 - 8
            while left_txt and _pil_measure(left_txt, size=11, face="jb", weight=400)[0] > avail:
                left_txt = left_txt[:-1]
            _pil_label(canvas, left_txt, L, y, C_TEXT, size=11, face="jb", weight=400)
            _pil_label(canvas, sev3, R-sw3, y, sc3, size=11, face="jb", weight=400)
            y += 12
            cv2.line(canvas, (L, y), (R, y), C_LINE, 1)
            y += 4

    _pil_label(canvas, "22SEAI64  |  MUSE, UoM", L, CANVAS_H-BOT_BAR_H-6, C_MIST, size="xs")


# ── Neural Load Strip — horizontal spectrum across full video width ───────────
def draw_neural_strip(canvas, video_y=None):
    """Spectrum filling the gap between top bar and video. video_y = v_off_y from main."""
    y0  = TOP_BAR_H
    x0  = 0
    x1  = VIDEO_W   # 1216px

    # Fill from y0 to wherever the video actually starts; fallback = 24px
    H = (video_y - y0) if (video_y is not None and video_y > y0 + 6) else 24
    if H < 6:
        return

    # Background + bottom border
    cv2.rectangle(canvas, (x0, y0), (x1, y0 + H), (7, 13, 20), -1)
    cv2.line(canvas, (x0, y0 + H - 1), (x1, y0 + H - 1), C_LINE, 1)

    ts = time.time()

    # Label rows sit at the bottom; bars fill everything above them
    lbl_h  = 12 if H >= 24 else 0
    pad_t  = 4
    pad_b  = lbl_h + 2
    bar_y0 = y0 + pad_t
    bar_h  = max(1, H - pad_t - pad_b)

    # Bars span the full width (no reserved label columns — labels overlay at bottom)
    NB    = 52
    b_gap = 2
    b_w   = max(1, (x1 - x0 - (NB - 1) * b_gap) // NB)

    peak_pct = 0.0
    peak_idx = 0
    bar_pcts = []
    for i in range(NB):
        hv  = 35 + math.sin(ts * 0.9 + i * 0.38) * 25 + math.sin(ts * 0.5 - i * 0.2) * 15
        pct = max(4.0, min(92.0, hv)) / 100.0
        bar_pcts.append(pct)
        if pct > peak_pct:
            peak_pct, peak_idx = pct, i

    for i, pct in enumerate(bar_pcts):
        bx     = x0 + i * (b_w + b_gap)
        bh_px  = max(1, int(pct * bar_h))
        by_top = bar_y0 + (bar_h - bh_px)
        if pct >= 0.70:
            bar_col = C_CYAN            # bright ambient red (peak)
        elif pct >= 0.40:
            bar_col = ( 40,  80, 220)   # mid-level orange-red
        else:
            bar_col = ( 25,  20, 110)   # low-level dark maroon
        for row in range(bh_px):
            t_g       = row / max(bh_px - 1, 1)
            intensity = 0.30 + 0.70 * t_g
            c   = (int(bar_col[0] * intensity),
                   int(bar_col[1] * intensity),
                   int(bar_col[2] * intensity))
            ry  = by_top + row
            if 0 <= ry < canvas.shape[0]:
                canvas[ry, bx:bx + b_w] = c
        if i == peak_idx and 0 <= by_top < canvas.shape[0]:
            canvas[by_top, bx:bx + b_w] = (220, 230, 240)

    # Labels overlaid at the bottom of the strip
    if lbl_h > 0:
        lbl_y = y0 + H - 3
        _pil_label(canvas, "NEURAL LOAD", x0 + 6, lbl_y, C_MIST, size=9, face="jb", weight=500)
        peak_str = f"PEAK {int(peak_pct * 100)}%"
        pw, _    = _pil_measure(peak_str, size=9, face="jb", weight=500)
        _pil_label(canvas, peak_str, x1 - pw - 6, lbl_y, C_CYAN, size=9, face="jb", weight=500)


# ── Top bar — header 68px, matches HTML .header grid row ──────────────────────
def draw_top_bar(canvas, frame_no, video_fps, session_start):
    h = TOP_BAR_H  # 68px, spec: grid-template-rows: 68px 1fr 34px

    # Background — dark panel, 1px border on bottom (C_LINE_FULL = full cyan)
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, h), (8, 11, 16), -1)
    cv2.line(canvas, (0, h - 1), (CANVAS_W, h - 1), C_LINE_FULL, 1)

    # ── Brand-mark: "CORVUS" — Orbitron 900 42px, vertically centred in 68px bar
    title      = "CORVUS"
    tx         = 22
    ink_w, ink_h = _pil_measure(title, size=42, face="orb", weight=900)
    if ink_w == 0: ink_w, ink_h = 200, 30
    ty = h // 2 + ink_h // 2
    _pil_label_glow(canvas, title, tx, ty, C_CYAN, size_px=42, weight=900, glow_r=14)
    tw = ink_w

    # ── Brand subtitle — JetBrains Mono, right of separator ─────────────────
    sep_x = tx + tw + 16
    cv2.line(canvas, (sep_x - 8, 10), (sep_x - 8, h - 10), C_LINE_FULL, 1)
    sub_x = sep_x + 4
    # l1: 14px bold
    _pil_label(canvas, "COLLISION ONSET RECOGNITION",         sub_x, 31, C_TEXT,   size=14, face="jb", weight=800)
    # l2: 12px bold violet
    _pil_label(canvas, "VEHICULAR · UNIFIED · SURVEILLANCE", sub_x, 51, C_VIOLET, size=12, face="jb", weight=800)

    # ── Mode pill — Orbitron 11px badge, cyan border, pulsing dot ────────────
    mode    = _state["mode"]
    trt_tag = " · TRT FP16" if _state.get("use_tensorrt") else ""
    badge   = f"{mode} MODE{trt_tag}"
    bw2, _  = _pil_measure(badge, size="sm", face="orb", weight=700)
    if bw2 == 0: bw2 = 80
    pill_cx = sub_x + (VIDEO_W - sub_x) // 2
    bx2     = pill_cx - (bw2 + 36) // 2

    cv2.rectangle(canvas, (bx2, 18), (bx2 + bw2 + 36, 50), (6, 9, 12), -1)
    cv2.rectangle(canvas, (bx2, 18), (bx2 + bw2 + 36, 50), C_LINE_FULL, 1)
    dot_col = _pulse_color(C_CYAN, 1.6)
    cv2.circle(canvas, (bx2 + 12, 34), 3, dot_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx2 + 12, 34), 5, tuple(int(c*0.25) for c in C_CYAN), 1)
    _pil_label(canvas, badge, bx2 + 22, 39, C_CYAN, size="sm", face="orb", weight=700)

    # ── Timers — JetBrains Mono 16px, right-aligned ───────────────────────────
    total_sec = int(frame_no / video_fps) if video_fps > 0 else 0
    ts_val    = f"{total_sec//3600:02d}:{(total_sec%3600)//60:02d}:{total_sec%60:02d}"
    elapsed   = int(time.time() - session_start)
    sess_val  = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"

    sv_w, _ = _pil_measure(sess_val, size=16, face="jb", weight=600)
    tv_w, _ = _pil_measure(ts_val,   size=16, face="jb", weight=600)
    if sv_w == 0: sv_w = 78
    if tv_w == 0: tv_w = 78
    sess_rx = VIDEO_W - 16
    vid_rx  = sess_rx - sv_w - 32

    cv2.line(canvas, (vid_rx - tv_w - 14, 10), (vid_rx - tv_w - 14, h - 10), C_LINE_FULL, 1)
    _pil_label(canvas, "VID",     vid_rx - tv_w - 2, 26, C_MIST, size="xs", face="jb", weight=400)
    _pil_label(canvas, ts_val,    vid_rx - tv_w,     50, C_TEXT, size=16,   face="jb", weight=600)
    _pil_label(canvas, "SESSION", sess_rx - sv_w,    26, C_MIST, size="xs", face="jb", weight=400)
    _pil_label(canvas, sess_val,  sess_rx - sv_w,    50, C_TEXT, size=16,   face="jb", weight=600)

    # ── Panel column border ───────────────────────────────────────────────────
    cv2.line(canvas, (PANEL_X, 0), (PANEL_X, h), C_LINE_FULL, 1)


# ── Bottom bar — 34px footer, matches HTML .footer ────────────────────────────
def draw_bottom_bar(canvas, accident_active, last_event=None):
    y0 = TOP_BAR_H + VIDEO_H   # 68 + 858 = 926
    h  = BOT_BAR_H             # 34

    cv2.rectangle(canvas, (0, y0), (CANVAS_W, y0 + h), (6, 8, 12), -1)
    cv2.line(canvas, (0, y0), (CANVAS_W, y0), C_LINE_FULL, 1)

    if accident_active:
        # ── Expanded collision banner — video width only, taller ─────────────
        BH    = 120                         # total banner height
        by0   = y0 - (BH - h)              # banner top
        pulse = _pulse_color(C_CRIMSON, 5.0)

        # Fill: video width only (0 → VIDEO_W)
        ov = canvas.copy()
        cv2.rectangle(ov, (0, by0), (VIDEO_W, y0 + h), (14, 2, 4), -1)
        cv2.addWeighted(ov, 0.92, canvas, 0.08, 0, canvas)

        # Pulsing top border — video width only
        cv2.line(canvas, (0, by0),     (VIDEO_W, by0),     pulse, 2)
        cv2.line(canvas, (0, by0 + 1), (VIDEO_W, by0 + 1),
                 tuple(int(c * 0.3) for c in C_CRIMSON), 1)

        # ── Row 1: "COLLISION DETECTED" Orbitron 28px centered ──────────────
        msg  = "COLLISION DETECTED"
        mw, _ = _pil_measure(msg, size=28, face="orb", weight=900)
        if mw == 0: mw = 380
        cx   = (VIDEO_W - mw) // 2
        _pil_label_glow(canvas, msg, cx, by0 + 56, pulse,
                        size_px=28, weight=900, glow_r=14)
        # Warning triangles flanking the text
        tri_y = by0 + 28
        for tx in (cx - 34, cx + mw + 14):
            pts = np.array([[tx, tri_y + 20], [tx + 12, tri_y], [tx + 24, tri_y + 20]], np.int32)
            cv2.fillPoly(canvas, [pts], pulse)
            cv2.putText(canvas, "!", (tx + 7, tri_y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (8, 2, 2), 1, cv2.LINE_AA)

        # ── Row 2: event detail — 14px bold JetBrains Mono ──────────────────
        if last_event is not None:
            ids_str   = " ".join(str(i) for i in last_event.vehicle_ids)
            flags_str = "  |  ".join(last_event.trigger_flags)
            sev_str   = f"{last_event.severity}  {last_event.severity_score:.2f}"
            detail    = f"IDs: {ids_str}    {sev_str}    {flags_str}"
            dw, _     = _pil_measure(detail, size=14, face="jb", weight=700)
            if dw == 0: dw = 400
            _pil_label(canvas, detail, (VIDEO_W - dw) // 2, by0 + 92,
                       C_TEXT, size=14, face="jb", weight=700)
    else:
        # HTML .footer: three segs; green dot + 11px JetBrains Mono mist
        segs = ["MONITORING ACTIVE", "ALL SYSTEMS NOMINAL", "NO INCIDENTS DETECTED"]
        sx   = 22
        for seg in segs:
            dot_col2 = _pulse_color(C_GREEN, 1.6)
            cv2.circle(canvas, (sx, y0 + 17), 3, dot_col2, -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, y0 + 17), 5, tuple(int(c*0.2) for c in C_GREEN), 1)
            sx += 11
            sw3, _ = _pil_measure(seg, size=13, face="jb", weight=600)
            _pil_label(canvas, seg, sx, y0 + 25, C_TEXT, size=13, face="jb", weight=600)
            sx += (sw3 if sw3 > 0 else 140) + 26

    cv2.line(canvas, (PANEL_X, y0), (PANEL_X, y0 + h), C_LINE_FULL, 1)


# ── Accident overlay — compact top banner ────────────────────────────────────
def draw_accident_overlay(frame, event, flash_frame=0):
    h, w = frame.shape[:2]

    # Corner vignette
    vign    = frame.copy()
    corners = [
        np.array([[0, 0],      [w//4, 0],    [0, h//4]],   np.int32),
        np.array([[w, 0],      [3*w//4, 0],  [w, h//4]],   np.int32),
        np.array([[0, h],      [w//4, h],    [0, 3*h//4]], np.int32),
        np.array([[w, h],      [3*w//4, h],  [w, 3*h//4]], np.int32),
    ]
    for pts in corners:
        cv2.fillPoly(vign, [pts], (0, 0, 0))
    cv2.addWeighted(vign, 0.18, frame, 0.82, 0, frame)

    # Scanline flicker first 5 frames
    if flash_frame < 5:
        scan = frame.copy()
        for yy in range(0, h, 5):
            cv2.line(scan, (0, yy), (w, yy), (0, 0, 0), 1)
        cv2.addWeighted(scan, 0.20, frame, 0.80, 0, frame)

    # Banner
    banner_h = 80
    ov       = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, banner_h), (5, 5, 35), -1)
    cv2.addWeighted(ov, 0.92, frame, 0.08, 0, frame)

    border_col = _pulse_color(C_RED, 4.0)
    cv2.line(frame, (0, banner_h), (w, banner_h), border_col, 2)
    cv2.line(frame, (0, 0),        (w, 0),        border_col, 1)

    ids_str   = " ".join(str(i) for i in event.vehicle_ids)
    flags_str = " + ".join(event.trigger_flags)
    zone_str  = f"  |  ZONE: {event.zone_id}" if event.zone_id else ""
    sev_str   = f"{event.severity} ({event.severity_score:.2f})"
    title     = f"COLLISION DETECTED  IDs:{ids_str}  {sev_str}{zone_str}"

    title_col = _pulse_color(C_RED, 4.0)
    tw, _ = _pil_measure(title, size=13, face="orb", weight=700)
    if tw == 0: tw = 300
    _pil_label(frame, title, max(10, (w - tw) // 2), 38, title_col, size=13, face="orb", weight=700)

    sw, _ = _pil_measure(flags_str, size=10, face="jb", weight=400)
    if sw == 0: sw = 200
    _pil_label(frame, flags_str, max(10, (w - sw) // 2), 66, C_AMBER, size=10, face="jb", weight=400)

    return frame


# ── Zone event banner ─────────────────────────────────────────────────────────
def draw_zone_event_banner(frame, zone_event):
    h, w = frame.shape[:2]
    col  = _pulse_color(C_AMBER, 2.0)
    ov   = frame.copy()
    cv2.rectangle(ov, (0, h-28), (w, h), (10, 30, 50), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, h-28), (w, h-28), col, 1)
    txt = (f"  [ZONE]  {zone_event.event_type}  |  "
           f"{zone_event.zone_name}  |  TRACK:{int(zone_event.track_id):02d}")
    cv2.putText(frame, txt, (8, h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


# ── Screenshot ────────────────────────────────────────────────────────────────
def save_screenshot(frame, event, screenshots_dir):
    os.makedirs(screenshots_dir, exist_ok=True)
    ts    = time.strftime("%Y%m%d_%H%M%S")
    ids   = "_".join(str(i) for i in event.vehicle_ids)
    fname = f"accident_{ts}_IDs_{ids}_{event.severity}.jpg"
    path  = os.path.join(screenshots_dir, fname)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.info(f"[Utils] Screenshot -> {path}")
    print(f"[Utils] Screenshot -> {path}")
    return path


# ── FPS counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=30):
        self._times  = []
        self._window = window

    def tick(self):
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times)-1) / (self._times[-1]-self._times[0])


# ── Legacy stubs ──────────────────────────────────────────────────────────────
def draw_hud(frame, fps, frame_no, n_tracks, mode="SEG", session_start=0): pass
def draw_video_timestamp(frame, frame_no, fps, session_start=0): pass
