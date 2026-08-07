# =============================================================================
# main.py — CORVUS v4
# =============================================================================
import argparse, cv2, time, logging, sys, os, threading
import numpy as np
import config
from detector       import VehicleDetector, Detection
from tracker        import CentroidTracker
from accident_logic import AccidentDetector
from alert          import TelegramAlerter
from zones          import ZoneManager
from logger         import EventLogger
from edge_config    import apply_edge_settings, export_openvino
from behavior       import BehaviorClassifier
from replay         import ReplayBuffer
from scene_cut      import SceneCutDetector
from utils import (
    draw_track, draw_accident_overlay, draw_zone_event_banner,
    draw_trajectory_predictions, draw_near_miss_highlights,
    draw_zone_dwell_overlays, draw_collision_effects, trigger_collision_fx,
    update_heatmap, draw_heatmap_overlay, compute_frame_data,
    save_screenshot, draw_side_panel, draw_top_bar, draw_bottom_bar,
    draw_feed_grid, draw_feed_scanline, draw_feed_vignette,
    draw_radar, draw_filmstrip, draw_accident_banner, draw_neural_strip,
    compute_ttc,
    FPSCounter, update_state, push_event,
    CANVAS_W, CANVAS_H, VIDEO_W, VIDEO_H, PANEL_X, TOP_BAR_H,
)

try:
    import winsound as _ws
    _WINSOUND = True
except ImportError:
    _WINSOUND = False

def _beep():
    try:
        _ws.Beep(880, 300)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",       type=str, default=None)
    p.add_argument("--start-frame", type=int, default=0,
                   help="Seek to this frame before processing starts (for debugging a specific segment)")
    p.add_argument("--box-only",    action="store_true")
    p.add_argument("--edge",        action="store_true")
    p.add_argument("--no-deepsort", action="store_true")
    p.add_argument("--export-ov",   action="store_true")
    p.add_argument("--export-trt",  action="store_true",
                   help="Export YOLOv8s-seg to TensorRT FP16 and exit")
    return p.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
        logging.FileHandler(os.path.join(config.LOG_DIR, "accident_detection.log"), mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


def main():
    # Must happen before any cv2 window is created — setting DPI awareness
    # after a window exists can reset it out of fullscreen on Windows.
    _screen_w = _screen_h = None
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
            _screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            _screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            _screen_w = _screen_h = None

    args = parse_args()

    if args.export_trt:
        from tensorrt_export import main as trt_main
        trt_main(); sys.exit(0)

    if args.edge:        apply_edge_settings(config)
    if args.box_only:    config.USE_BOX_ONLY = True
    if args.no_deepsort: config.USE_DEEPSORT = False
    if args.video:       config.VIDEO_PATH   = args.video
    if args.export_ov:
        export_openvino(config.YOLO_BOX_MODEL_PATH); sys.exit(0)

    log.info("=" * 60)
    log.info("CORVUS v4 — Traffic Accident Detection System")
    log.info("=" * 60)

    if not os.path.isfile(config.VIDEO_PATH):
        print(f"\nVideo not found: {config.VIDEO_PATH}\n"); sys.exit(1)

    print("\n[Main] Initialising subsystems ...")
    detector      = VehicleDetector(config)
    tracker       = CentroidTracker(config)
    zones         = ZoneManager(config.ZONES_FILE)
    event_log     = EventLogger(config.LOG_DIR)
    alerter       = TelegramAlerter(config)
    behavior_clf  = BehaviorClassifier()
    cut_detector  = SceneCutDetector(config)
    fps_ctr       = FPSCounter(window=30)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        log.error(f"Cannot open: {config.VIDEO_PATH}"); sys.exit(1)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay  = max(1, int(1000 / config.TARGET_FPS))

    replay_buf = ReplayBuffer(
        fps         = video_fps,
        pre_seconds = config.REPLAY_PRE_SECONDS,
        post_seconds= config.REPLAY_POST_SECONDS,
    )

    scale   = min(VIDEO_W / src_w, VIDEO_H / src_h)
    disp_w  = int(src_w * scale)
    disp_h  = int(src_h * scale)
    _crop_x = 0
    v_off_y = TOP_BAR_H + (VIDEO_H - disp_h) // 2
    v_off_x = (VIDEO_W - disp_w) // 2

    log.info(f"Video: {config.VIDEO_PATH} | {video_fps:.1f} FPS | {total_frames} frames")
    log.info(f"Display: {disp_w}x{disp_h} | TensorRT: {config.USE_TENSORRT}")

    accident      = AccidentDetector(config, frame_size=(disp_w, disp_h))
    mode_str      = "BOX" if (config.USE_BOX_ONLY or config.EDGE_MODE) else "SEG"
    session_start = time.time()

    update_state(
        mode             = mode_str,
        session_start    = session_start,
        video_fps        = video_fps,
        pixels_per_meter = config.PIXELS_PER_METER,
        use_tensorrt     = config.USE_TENSORRT,
    )

    headless = bool(os.environ.get("CORVUS_HEADLESS"))
    if not headless:
        cv2.namedWindow("CORVUS", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("CORVUS", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── Pre-scale to the real screen resolution ──────────────────────────────
    # cv2's own fullscreen stretch uses low-quality (nearest/bilinear) scaling,
    # which smears small sidebar text at high DPI and can make tightly-spaced
    # rows look like they're overlapping even though the source canvas isn't.
    # Doing our own high-quality resize + letterbox makes the window's stretch
    # a no-op and keeps text crisp at whatever the monitor's real resolution is.
    disp_scale, disp_ox, disp_oy = 1.0, 0, 0
    screen_w, screen_h = _screen_w, _screen_h
    if screen_w and screen_h:
        disp_scale = min(screen_w / CANVAS_W, screen_h / CANVAS_H)
        if disp_scale <= 1.01:
            disp_scale, screen_w, screen_h = 1.0, None, None   # no upscale needed
        else:
            disp_ox = (screen_w - int(CANVAS_W * disp_scale)) // 2
            disp_oy = (screen_h - int(CANVAS_H * disp_scale)) // 2

    print(f"\n[Main] CORVUS ready | Mode:{mode_str} | TRT:{config.USE_TENSORRT}")
    print(f"[Main] Keys: Q=quit  SPACE=pause  S=screenshot  Z=zones  H=heatmap\n")

    frame_no        = args.start_frame
    paused          = False
    accident_flash  = 0
    flash_frame     = 0
    FLASH_DURATION  = 60
    last_event      = None
    show_zones      = False
    show_heatmap    = False
    fps_history     = []
    accident_frames = []   # frame numbers where an accident was active (for filmstrip)
    canvas          = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    while True:
        key = -1 if headless else (cv2.waitKey(frame_delay) & 0xFF)
        if key in (ord('q'), 27):
            log.info("User exit."); break
        if key == ord(' '):
            paused = not paused
            print(f"[Main] {'PAUSED' if paused else 'RESUMED'}")
        if key == ord('z'):
            show_zones = not show_zones
        if key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"[Main] Heatmap {'ON' if show_heatmap else 'OFF'}")
        if key == ord('s') and last_event:
            save_screenshot(
                canvas[v_off_y:v_off_y+disp_h, v_off_x:v_off_x+disp_w],
                last_event, config.SCREENSHOTS_DIR,
            )
        if paused:
            time.sleep(0.03); continue

        ret, frame = cap.read()
        if not ret:
            log.info("End of video."); break

        if cut_detector.check(frame):
            tracker.reset()
            accident.notify_scene_cut()
            replay_buf.reset()
            zones.reset()
            log.info(f"[Main] Scene cut at frame {frame_no+1} — tracker & confirmation state reset")

        frame_no += 1
        fps = fps_ctr.tick()
        fps_history.append(fps)

        display_frame = cv2.resize(frame, (disp_w, disp_h))

        # ── Detect ────────────────────────────────────────────────────────────
        detections = detector.detect(frame)
        sx, sy     = disp_w / src_w, disp_h / src_h
        scaled     = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            sbbox = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
            smask = ([[int(p[0]*sx), int(p[1]*sy)] for p in det.mask_contour]
                     if det.mask_contour else None)
            scaled.append(Detection(bbox=sbbox, label=det.label,
                                    confidence=det.confidence,
                                    class_id=det.class_id, mask_contour=smask))

        # ── Track ─────────────────────────────────────────────────────────────
        tracks = tracker.update(scaled, display_frame)

        # ── Zone update ───────────────────────────────────────────────────────
        zone_events = zones.update(tracks)
        for ze in zone_events:
            event_log.log_zone_event(ze, frame_no)

        # ── Frame analytics ───────────────────────────────────────────────────
        frame_data = compute_frame_data(tracks,
                                        near_miss_dist=getattr(config, "NEAR_MISS_DIST_PX", 80))
        behaviors  = {t.id: behavior_clf.classify(t, config) for t in tracks}

        # Zone dwell progress for overlay bars
        zone_dwells = {}
        for zone in zones.zones:
            if zone.loiter_sec > 0 and zone._entry_times:
                now = time.time()
                for tid, entry_t in zone._entry_times.items():
                    pct = min(1.0, (now - entry_t) / zone.loiter_sec)
                    zone_dwells.setdefault(zone.name, []).append((tid, pct))

        # ── Accident analysis ─────────────────────────────────────────────────
        acc_events = accident.analyse(tracks, zones.zones)

        # ── Handle accidents BEFORE drawing so FX activates this frame ────────
        involved_ids = set()
        for event in acc_events:
            involved_ids.update(event.vehicle_ids)

            col_tracks = [t for t in tracks if t.id in set(event.vehicle_ids)]
            if col_tracks:
                avg_cx = sum(t.current_centroid()[0] for t in col_tracks) // len(col_tracks)
                avg_cy = sum(t.current_centroid()[1] for t in col_tracks) // len(col_tracks)
            else:
                avg_cx, avg_cy = disp_w // 2, disp_h // 2

            trigger_collision_fx(event, avg_cx, avg_cy)
            replay_buf.trigger(event, config.REPLAY_DIR)

            if getattr(config, "AUDIO_ALERT", True) and _WINSOUND:
                threading.Thread(target=_beep, daemon=True).start()

            log.warning(
                f"ACCIDENT | IDs={event.vehicle_ids} | "
                f"{event.trigger_flags} | {event.severity} ({event.severity_score})"
                + (f" | Zone:{event.zone_id}" if event.zone_id else "")
            )
            print(f"\n{'='*54}")
            print(f"  ACCIDENT @ frame {frame_no}")
            print(f"  IDs      : {event.vehicle_ids}")
            print(f"  Triggers : {event.trigger_flags}")
            print(f"  Severity : {event.severity} ({event.severity_score})")
            if event.zone_id:
                print(f"  Zone     : {event.zone_id}")
            print(f"{'='*54}\n")

            push_event(event)
            alert_frame     = draw_accident_overlay(display_frame.copy(), event, 0)
            screenshot_path = save_screenshot(alert_frame, event, config.SCREENSHOTS_DIR)
            event_log.log_accident(event, frame_no, screenshot_path)
            alerter.send(screenshot_path, event)
            accident_flash = FLASH_DURATION
            flash_frame    = 0
            last_event     = event

        # ── Compute TTC ───────────────────────────────────────────────────────
        ttc_sec, ttc_pair, ttc_delta_kmh = compute_ttc(tracks)

        # ── Update shared state ───────────────────────────────────────────────
        n_vehicles    = sum(1 for t in tracks if t.label != "person")
        n_pedestrians = sum(1 for t in tracks if t.label == "person")
        active_zone_names = list({ze.zone_name for ze in zone_events})
        zone_status = [
            {"name": zone.name, "occupied": len(zone._entry_times)}
            for zone in zones.zones
        ]
        update_state(
            fps           = fps,
            frame_no      = frame_no,
            n_vehicles    = n_vehicles,
            n_pedestrians = n_pedestrians,
            n_tracks      = len(tracks),
            active_zones  = active_zone_names,
            zone_status   = zone_status,
            ttc_sec       = ttc_sec,
            ttc_pair      = ttc_pair,
            ttc_delta_kmh = ttc_delta_kmh,
            heatmap_on    = show_heatmap,
        )

        # ── Draw ──────────────────────────────────────────────────────────────
        # Feed overlays (applied first, underneath detections)
        draw_feed_grid(display_frame)
        draw_feed_scanline(display_frame, frame_no)

        if show_heatmap:
            update_heatmap(tracks, display_frame)
            draw_heatmap_overlay(display_frame)

        if show_zones:
            zones.draw(display_frame)
            draw_zone_dwell_overlays(display_frame, zones.zones, zone_dwells)

        draw_trajectory_predictions(display_frame, tracks)

        seg = detector.uses_segmentation
        for track in tracks:
            draw_track(
                display_frame, track,
                accident   = (track.id in involved_ids),
                seg_mode   = seg,
                behavior   = behaviors.get(track.id, "NORMAL"),
                risk_score = frame_data["risk_score"],
                frame_no   = frame_no,
            )

        draw_near_miss_highlights(display_frame, frame_data["near_miss_pairs"])

        # Collision effects: shockwave, zoom, flash (FX was set above)
        draw_collision_effects(display_frame)

        # Track flash counter (overlay removed — info shown in bottom banner only)
        if accident_flash > 0:
            accident_flash -= 1
            flash_frame    += 1
            if frame_no not in accident_frames:
                accident_frames.append(frame_no)
                if len(accident_frames) > 60:
                    accident_frames.pop(0)

        if zone_events:
            draw_zone_event_banner(display_frame, zone_events[-1])

        # Post-detection overlays (drawn on top of tracks)
        draw_radar(display_frame, tracks, disp_h, disp_w)
        draw_filmstrip(display_frame, frame_no, accident_frames)
        if accident_flash > 0 and last_event:
            draw_accident_banner(display_frame, last_event)

        # Add frame to replay buffer (after all drawing, before canvas compose)
        replay_buf.add_frame(display_frame)

        # ── Compose canvas ────────────────────────────────────────────────────
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[v_off_y:v_off_y+disp_h, v_off_x:v_off_x+disp_w] = display_frame
        draw_top_bar(canvas, frame_no, video_fps, session_start)
        draw_neural_strip(canvas, v_off_y)
        draw_side_panel(canvas)
        draw_bottom_bar(canvas, accident_flash > 0, last_event)

        if os.environ.get("CORVUS_DEBUG_DUMP") and frame_no % 15 == 0:
            os.makedirs("debug_dump", exist_ok=True)
            cv2.imwrite(f"debug_dump/frame_{frame_no:05d}.png", canvas)

        if not headless:
            if screen_w and screen_h:
                scaled_w = int(CANVAS_W * disp_scale)
                scaled_h = int(CANVAS_H * disp_scale)
                resized  = cv2.resize(canvas, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
                out      = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                out[disp_oy:disp_oy+scaled_h, disp_ox:disp_ox+scaled_w] = resized
                cv2.imshow("CORVUS", out)
            else:
                cv2.imshow("CORVUS", canvas)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
    log.info("Shutting down ...")
    cap.release()
    replay_buf.close()
    cv2.destroyAllWindows()
    alerter.stop()
    event_log.close(avg_fps=avg_fps, total_frames=frame_no)
    print(f"\n[Main] Done. Screenshots → {config.SCREENSHOTS_DIR}")
    print(f"[Main] Logs       → {config.LOG_DIR}")
    print(f"[Main] Replays    → {config.REPLAY_DIR}")


if __name__ == "__main__":
    main()
