"""
Driver Drowsiness Detection System  ─  Enhanced v3
=====================================================
Detection pipeline:
  ✓ Eye-Aspect Ratio (EAR)  +  PERCLOS rolling score
  ✓ Mouth-Aspect Ratio (MAR)  — yawn detection
  ✓ Head-nod tracking (nose-tip Y displacement)
  ✓ Blink counter  (blinks per minute)
  ✓ Micro-sleep timer  (eyes continuously closed)
  ✓ Session timer  (total driving time displayed)
  ✓ Drowsiness event log  (saved to drowsiness_log.csv)
  ✓ System-speaker audio alarm  (winsound, background thread)

Controls:  F = toggle fullscreen │ Esc = quit
"""

import sys, os, time, threading, collections, csv, datetime
import winsound
import numpy as np
import dlib
import cv2
from math import hypot

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DAT_FILE            = "shape_predictor_68_face_landmarks.dat"
CAMERA_INDEX        = 0
LOG_FILE            = "drowsiness_log.csv"

# ── Thresholds ─────────────────────────────────────────────────
EAR_CLOSED_THRESH   = 4.30    # EAR ≥ this → eyes closed
PERCLOS_WINDOW_S    = 2.5     # rolling window for PERCLOS
PERCLOS_THRESH      = 0.30    # > 30 % closed in window → drowsy
MAR_YAWN_THRESH     = 0.38    # MAR ≥ this → yawning
NOD_DELTA_Y         = 18      # nose-drop (px) to register a nod
NOD_COUNT_THRESH    = 3       # nods needed in NOD_WINDOW_S
NOD_WINDOW_S        = 3.0
MICROSLEEP_S        = 2.0     # continuous eye-close → micro-sleep
BLINK_CLOSE_FRAMES  = 2       # frames to confirm a blink-close
BLINK_OPEN_FRAMES   = 2       # frames to confirm re-open
BPM_WINDOW_S        = 60.0    # blink-rate window

# ── Performance ────────────────────────────────────────────────
FRAME_DELAY_MS      = 1       # ≈ native speed (no artificial lag)
UPSAMPLE            = 0       # 0 = fast, 1 = detects smaller faces
SMOOTH_ALPHA        = 0.25    # EMA factor — higher = more responsive
STATUS_HOLD_FRAMES  = 15      # frames to hold a non-AWAKE status

# ── Alert ──────────────────────────────────────────────────────
ALERT_COOLDOWN_S    = 5.0

# ── Colours (BGR) ──────────────────────────────────────────────
C_GREEN  = ( 80, 215,  90)
C_RED    = ( 50,  50, 220)
C_ORANGE = ( 30, 160, 255)
C_YELLOW = ( 40, 210, 210)
C_BLUE   = (220, 160,  40)
C_WHITE  = (230, 230, 230)
C_GREY   = (140, 140, 140)
C_DARK   = ( 25,  25,  25)
C_PANEL  = ( 35,  35,  35)
C_BORDER = ( 70,  70,  70)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════
def _mid(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def ear(idx, lm):
    lp = (lm.part(idx[0]).x, lm.part(idx[0]).y)
    rp = (lm.part(idx[3]).x, lm.part(idx[3]).y)
    ct = _mid(lm.part(idx[1]), lm.part(idx[2]))
    cb = _mid(lm.part(idx[5]), lm.part(idx[4]))
    h = hypot(lp[0]-rp[0], lp[1]-rp[1])
    v = hypot(ct[0]-cb[0], ct[1]-cb[1])
    return (h / v) if v else 0.0

def mar(idx, lm):
    lp  = (lm.part(idx[0]).x, lm.part(idx[0]).y)
    rp  = (lm.part(idx[2]).x, lm.part(idx[2]).y)
    top = (lm.part(idx[1]).x, lm.part(idx[1]).y)
    bot = (lm.part(idx[3]).x, lm.part(idx[3]).y)
    h = hypot(lp[0]-rp[0], lp[1]-rp[1])
    v = hypot(top[0]-bot[0], top[1]-bot[1])
    return (v / h) if h else v

# ═══════════════════════════════════════════════════════════════
#  ALERT ENGINE
# ═══════════════════════════════════════════════════════════════
class AlertEngine:
    def __init__(self):
        self._lock, self._active = threading.Lock(), False
        self._last, self._stop   = 0.0, threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            winsound.Beep(1200, 280)
            time.sleep(0.12)
            if not self._stop.is_set():
                winsound.Beep(900, 280)
            time.sleep(0.08)

    def trigger(self):
        now = time.time()
        with self._lock:
            if self._active or now - self._last < ALERT_COOLDOWN_S:
                return
            self._last, self._active = now, True
            self._stop.clear()
            threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        with self._lock:
            if not self._active: return
            self._stop.set(); self._active = False

    @property
    def is_active(self): return self._active

# ═══════════════════════════════════════════════════════════════
#  PERCLOS TRACKER
# ═══════════════════════════════════════════════════════════════
class PerclosTracker:
    def __init__(self):
        self._buf = collections.deque()

    def update(self, ear_val):
        now = time.time()
        self._buf.append((now, ear_val >= EAR_CLOSED_THRESH))
        cutoff = now - PERCLOS_WINDOW_S
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    @property
    def score(self):
        if not self._buf: return 0.0
        return sum(1 for _, c in self._buf if c) / len(self._buf)

# ═══════════════════════════════════════════════════════════════
#  HEAD-NOD TRACKER
# ═══════════════════════════════════════════════════════════════
class NodTracker:
    def __init__(self):
        self._prev_y, self._times = None, collections.deque()

    def update(self, ny):
        now = time.time()
        if self._prev_y is not None and (ny - self._prev_y) > NOD_DELTA_Y:
            self._times.append(now)
        self._prev_y = ny
        cutoff = now - NOD_WINDOW_S
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    @property
    def count(self): return len(self._times)

    @property
    def alarming(self): return self.count >= NOD_COUNT_THRESH

# ═══════════════════════════════════════════════════════════════
#  BLINK TRACKER  (new)
# ═══════════════════════════════════════════════════════════════
class BlinkTracker:
    """Counts blinks per minute using a simple state machine."""
    def __init__(self):
        self._closed_streak = 0
        self._open_streak   = 0
        self._in_blink      = False
        self._blink_times   = collections.deque()

    def update(self, is_closed):
        now = time.time()
        if not self._in_blink:
            if is_closed:
                self._closed_streak += 1
                self._open_streak    = 0
                if self._closed_streak >= BLINK_CLOSE_FRAMES:
                    self._in_blink = True
            else:
                self._closed_streak = 0
        else:
            if not is_closed:
                self._open_streak += 1
                if self._open_streak >= BLINK_OPEN_FRAMES:
                    self._blink_times.append(now)
                    self._in_blink      = False
                    self._closed_streak = 0
                    self._open_streak   = 0
            else:
                self._open_streak = 0

        cutoff = now - BPM_WINDOW_S
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

    @property
    def bpm(self):
        """Blinks per minute (scaled to a 60-s window)."""
        if not self._blink_times:
            return 0.0
        elapsed = time.time() - self._blink_times[0]
        if elapsed < 5.0:           # not enough data yet
            return 0.0
        return len(self._blink_times) / elapsed * 60.0

# ═══════════════════════════════════════════════════════════════
#  MICRO-SLEEP TRACKER  (new)
# ═══════════════════════════════════════════════════════════════
class MicroSleepTracker:
    """Measures how long eyes have been CONTINUOUSLY closed."""
    def __init__(self):
        self._close_start = None

    def update(self, is_closed):
        if is_closed:
            if self._close_start is None:
                self._close_start = time.time()
        else:
            self._close_start = None

    @property
    def duration(self):
        if self._close_start is None: return 0.0
        return time.time() - self._close_start

    @property
    def alarming(self):
        return self.duration >= MICROSLEEP_S

# ═══════════════════════════════════════════════════════════════
#  EVENT LOGGER  (new)
# ═══════════════════════════════════════════════════════════════
class EventLogger:
    def __init__(self, path=LOG_FILE):
        self._path = path
        self._last_event = {}
        # write header if file is new
        if not os.path.isfile(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "event", "ear", "perclos_pct", "mar",
                     "nod_count", "blink_bpm", "microsleep_s"])

    def log(self, event, ear_v, perclos, mar_v, nods, bpm, ms_dur):
        now = time.time()
        # de-duplicate: don't log same event more than once per 10 s
        if now - self._last_event.get(event, 0) < 10.0:
            return
        self._last_event[event] = now
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, event,
                f"{ear_v:.2f}", f"{perclos*100:.1f}",
                f"{mar_v:.2f}", nods,
                f"{bpm:.1f}", f"{ms_dur:.1f}"])

# ═══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════
def alpha_rect(img, x1, y1, x2, y2, colour, alpha=0.72):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def metric_bar(img, x, y, w, label, val_str, frac, col, bar_h=9, row_h=46):
    pad = 10
    cv2.putText(img, label,   (x+pad, y+17),   FONT, 0.46, C_GREY,  1, cv2.LINE_AA)
    (vw,_),_ = cv2.getTextSize(val_str, FONT_BOLD, 0.52, 1)
    cv2.putText(img, val_str, (x+w-vw-pad, y+17), FONT_BOLD, 0.52, col, 1, cv2.LINE_AA)
    bx, by, bw = x+pad, y+24, w-2*pad
    cv2.rectangle(img, (bx, by), (bx+bw, by+bar_h), (55,55,55), -1)
    fill = max(0, min(int(bw*frac), bw))
    if fill:
        cv2.rectangle(img, (bx, by), (bx+fill, by+bar_h), col, -1)
    cv2.rectangle(img, (bx, by), (bx+bw, by+bar_h), C_BORDER, 1)

def draw_sidebar(img, ear_s, perclos, mar_s, nod_c, bpm,
                 ms_dur, status, alert_on, session_s):
    H, W = img.shape[:2]
    SW   = 330
    sx   = W - SW

    # background
    alpha_rect(img, sx, 0, W, H, (22, 22, 22), 0.82)
    cv2.line(img, (sx, 0), (sx, H), C_BORDER, 1)

    y = 0

    # ── TITLE ──────────────────────────────────────────────────
    y += 32
    cv2.putText(img, "DROWSINESS", (sx+12, y), FONT_BOLD, 0.72, C_WHITE,  1, cv2.LINE_AA)
    y += 24
    cv2.putText(img, "DETECTOR",   (sx+12, y), FONT_BOLD, 0.72, C_YELLOW, 1, cv2.LINE_AA)
    y += 10
    cv2.line(img, (sx+8, y), (W-8, y), C_BORDER, 1)
    y += 8

    # ── STATUS BADGE ───────────────────────────────────────────
    status_cfg = {
        "AWAKE"       : (C_GREEN,  "Driver is alert"),
        "DROWSY"      : (C_RED,    "Eyes closing — stay alert!"),
        "YAWNING"     : (C_ORANGE, "Excessive yawning"),
        "NODDING"     : (C_YELLOW, "Head nodding detected"),
        "MICRO-SLEEP" : (C_RED,    f"Eyes closed {ms_dur:.1f}s — WAKE UP!"),
    }
    col, desc = status_cfg.get(status, (C_WHITE, ""))
    cx = sx + SW // 2

    # badge background
    alpha_rect(img, sx+10, y, W-10, y+68, (col[0]//6, col[1]//6, col[2]//6), 0.80)
    cv2.rectangle(img, (sx+10, y), (W-10, y+68), col, 1)

    (lw, lh), _ = cv2.getTextSize(status, FONT_BOLD, 1.05, 2)
    cv2.putText(img, status, (cx - lw//2 + 1, y+40+1), FONT_BOLD, 1.05, (0,0,0),    3, cv2.LINE_AA)
    cv2.putText(img, status, (cx - lw//2,     y+40  ), FONT_BOLD, 1.05, col,        2, cv2.LINE_AA)
    (sw,_),_ = cv2.getTextSize(desc, FONT, 0.44, 1)
    cv2.putText(img, desc, (cx - sw//2, y+62), FONT, 0.44, C_GREY, 1, cv2.LINE_AA)
    y += 76

    # ── METRIC BARS ────────────────────────────────────────────
    ear_col = C_RED if ear_s >= EAR_CLOSED_THRESH else C_GREEN
    eye_lbl = "Closed" if ear_s >= EAR_CLOSED_THRESH else "Open"
    metric_bar(img, sx, y, SW, f"EYES  [{eye_lbl}]",
               f"{ear_s:.2f}", min(ear_s/8.0,1.0), ear_col)
    y += 46

    perc_col = C_RED if perclos >= PERCLOS_THRESH else (C_YELLOW if perclos >= 0.15 else C_GREEN)
    metric_bar(img, sx, y, SW, f"PERCLOS  (>{int(PERCLOS_THRESH*100)}% = alert)",
               f"{perclos*100:.0f}%", perclos, perc_col)
    y += 46

    mar_col = C_ORANGE if mar_s >= MAR_YAWN_THRESH else C_GREEN
    metric_bar(img, sx, y, SW, "MOUTH / YAWN",
               "Yawning" if mar_s >= MAR_YAWN_THRESH else f"{mar_s:.2f}",
               min(mar_s/0.8,1.0), mar_col)
    y += 46

    nod_col = C_RED if nod_c >= NOD_COUNT_THRESH else C_WHITE
    metric_bar(img, sx, y, SW, f"HEAD NODS  (in {NOD_WINDOW_S:.0f}s)",
               f"{nod_c}/{NOD_COUNT_THRESH}", nod_c/NOD_COUNT_THRESH, nod_col)
    y += 46

    # ── NEW METRICS ────────────────────────────────────────────
    cv2.line(img, (sx+8, y), (W-8, y), C_BORDER, 1)
    y += 4

    # Blinks per minute
    bpm_col = C_RED if bpm < 8 or bpm > 30 else C_GREEN
    bpm_note = "Low (fatigue?)" if bpm < 8 else ("High" if bpm > 30 else "Normal")
    metric_bar(img, sx, y, SW, "BLINK RATE (BPM)",
               f"{bpm:.0f}  {bpm_note}", min(bpm/40.0, 1.0), bpm_col)
    y += 46

    # Micro-sleep duration
    ms_col   = C_RED if ms_dur >= MICROSLEEP_S else (C_ORANGE if ms_dur >= 1.0 else C_GREEN)
    ms_note  = f"{ms_dur:.1f}s" if ms_dur > 0 else "None"
    metric_bar(img, sx, y, SW, f"MICRO-SLEEP  (>{MICROSLEEP_S:.0f}s = alert)",
               ms_note, min(ms_dur/4.0, 1.0), ms_col)
    y += 46

    # Session timer
    h, rem   = divmod(int(session_s), 3600)
    m, s     = divmod(rem, 60)
    sess_str = f"{h:02d}:{m:02d}:{s:02d}"
    sess_col = C_RED if session_s > 7200 else (C_YELLOW if session_s > 3600 else C_WHITE)
    cv2.putText(img, "SESSION TIME", (sx+10, y+16), FONT, 0.46, C_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, sess_str, (sx+SW-85, y+16), FONT_BOLD, 0.56, sess_col, 1, cv2.LINE_AA)
    y += 28

    cv2.line(img, (sx+8, y), (W-8, y), C_BORDER, 1)
    y += 8

    # ── ALERT INDICATOR ────────────────────────────────────────
    if alert_on:
        alpha_rect(img, sx+8, y, W-8, y+40, (0,0,170), 0.85)
        cv2.rectangle(img, (sx+8, y), (W-8, y+40), C_RED, 2)
        cv2.putText(img, "  AUDIO ALARM ACTIVE", (sx+16, y+26),
                    FONT_BOLD, 0.54, (255,255,255), 1, cv2.LINE_AA)
    else:
        alpha_rect(img, sx+8, y, W-8, y+40, (0,60,0), 0.60)
        cv2.rectangle(img, (sx+8, y), (W-8, y+40), C_GREEN, 1)
        cv2.putText(img, "  System Active — No Alert", (sx+16, y+26),
                    FONT_BOLD, 0.50, C_GREEN, 1, cv2.LINE_AA)
    y += 48

    # ── LOG NOTE ───────────────────────────────────────────────
    cv2.putText(img, f"Log: {LOG_FILE}", (sx+10, y+14),
                FONT, 0.37, C_GREY, 1, cv2.LINE_AA)

    # ── CONTROLS ────────────────────────────────────────────────
    cv2.putText(img, "F = Fullscreen    Esc = Quit",
                (sx+10, H-12), FONT, 0.40, C_GREY, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    if not os.path.isfile(DAT_FILE):
        print(f"\n[ERROR] '{DAT_FILE}' not found.\n"
              "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open camera {CAMERA_INDEX}.\n")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Loading models …")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DAT_FILE)
    print("[INFO] Ready.  F = fullscreen  |  Esc = quit")

    alert_engine  = AlertEngine()
    perclos_tr    = PerclosTracker()
    nod_tr        = NodTracker()
    blink_tr      = BlinkTracker()
    microsleep_tr = MicroSleepTracker()
    logger        = EventLogger()

    ear_smooth  = 0.0
    mar_smooth  = 0.0
    status      = "AWAKE"
    hold_ctr    = 0
    is_fs       = False
    session_start = time.time()

    WIN = "Driver Drowsiness Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.moveWindow(WIN, 0, 0)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        H, W  = frame.shape[:2]
        SW    = 330
        canvas = np.zeros((H, W + SW, 3), dtype=np.uint8)
        canvas[:, :W] = frame

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, UPSAMPLE)

        raw_ear, raw_mar = 0.0, 0.0
        new_status       = "AWAKE"
        eyes_closed      = False

        for face_roi in faces:
            lm = predictor(gray, face_roi)

            l_ear   = ear([36,37,38,39,40,41], lm)
            r_ear   = ear([42,43,44,45,46,47], lm)
            raw_ear = (l_ear + r_ear) / 2

            raw_mar = (mar([60,62,64,66], lm) + mar([48,51,54,57], lm)) / 2

            perclos_tr.update(raw_ear)
            eyes_closed = raw_ear >= EAR_CLOSED_THRESH
            blink_tr.update(eyes_closed)
            microsleep_tr.update(eyes_closed)
            nod_tr.update(lm.part(30).y)

            # status priority: MICRO-SLEEP > DROWSY > NODDING > YAWNING > AWAKE
            if microsleep_tr.alarming:
                new_status = "MICRO-SLEEP"
            elif perclos_tr.score >= PERCLOS_THRESH:
                new_status = "DROWSY"
            elif nod_tr.alarming:
                new_status = "NODDING"
            elif raw_mar >= MAR_YAWN_THRESH:
                new_status = "YAWNING"

            # face bounding box
            x,  y  = face_roi.left(),  face_roi.top()
            x1, y1 = face_roi.right(), face_roi.bottom()
            box_col = C_GREEN if new_status == "AWAKE" else C_RED
            cv2.rectangle(canvas, (x, y), (x1, y1), box_col, 2)
            if new_status != "AWAKE":
                cv2.putText(canvas, new_status, (x, y-6),
                            FONT_BOLD, 0.58, box_col, 2, cv2.LINE_AA)

        # smooth metrics
        if faces:
            ear_smooth = SMOOTH_ALPHA * raw_ear + (1-SMOOTH_ALPHA) * ear_smooth
            mar_smooth = SMOOTH_ALPHA * raw_mar + (1-SMOOTH_ALPHA) * mar_smooth
        else:
            ear_smooth = max(0.0, ear_smooth - 0.05)
            mar_smooth = max(0.0, mar_smooth - 0.005)
            blink_tr.update(False)
            microsleep_tr.update(False)

        # status hold
        if new_status != "AWAKE":
            status   = new_status
            hold_ctr = STATUS_HOLD_FRAMES
        else:
            hold_ctr = max(0, hold_ctr - 1)
            if hold_ctr == 0:
                status = "AWAKE"

        # alert
        if status != "AWAKE":
            alert_engine.trigger()
        else:
            alert_engine.stop()

        # log events
        ms_dur = microsleep_tr.duration
        bpm    = blink_tr.bpm
        if status != "AWAKE":
            logger.log(status, ear_smooth, perclos_tr.score,
                       mar_smooth, nod_tr.count, bpm, ms_dur)

        session_s = time.time() - session_start

        draw_sidebar(canvas, ear_smooth, perclos_tr.score, mar_smooth,
                     nod_tr.count, bpm, ms_dur, status,
                     alert_engine.is_active, session_s)

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
        if key == 27:
            break
        elif key in (ord('f'), ord('F')):
            is_fs = not is_fs
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if is_fs else cv2.WINDOW_NORMAL)

    alert_engine.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended. Events saved to '{LOG_FILE}'.")


if __name__ == "__main__":
    main()
