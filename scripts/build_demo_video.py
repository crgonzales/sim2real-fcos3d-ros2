#!/usr/bin/env python3
"""Assemble the complete CMPE 249 demo video from slides + detection footage."""
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 960, 540, 10
OUT = 'docs/media/cmpe249_demo.mp4'
CLIP = 'docs/media/demo_fcos3d_nuscenes.mp4'

FD = '/System/Library/Fonts/Supplemental/'
def F(name, size):
    return ImageFont.truetype(FD + name, size)

BOLD, REG, MONO = 'Arial Bold.ttf', 'Arial.ttf', 'Courier New Bold.ttf'

BG      = (14, 17, 23)
FG      = (232, 237, 243)
DIM     = (139, 148, 158)
ACCENT  = (88, 166, 255)
GOOD    = (63, 185, 80)
WARN    = (210, 153, 34)
BAD     = (248, 81, 73)

def slide():
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 4], fill=ACCENT)
    return img, d

def to_frames(img, seconds):
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return [arr] * int(seconds * FPS)

def kicker(d, text):
    d.text((56, 40), text, font=F(BOLD, 15), fill=ACCENT)

frames = []

# ---------------------------------------------------------------- 1. title
img, d = slide()
d.text((56, 150), 'Applying and Evaluating', font=F(REG, 40), fill=DIM)
d.text((56, 198), 'a Pretrained Monocular', font=F(BOLD, 44), fill=FG)
d.text((56, 250), '3D Object Detector in ROS 2', font=F(BOLD, 44), fill=FG)
d.text((56, 330), 'FCOS3D  ·  ROS 2 Humble  ·  nuScenes', font=F(REG, 22), fill=ACCENT)
d.text((56, 420), 'Carlos Gonzales', font=F(BOLD, 24), fill=FG)
d.text((56, 452), 'CMPE 249  ·  Option 1 (Apply & Evaluate)  ·  Focused area: Application',
       font=F(REG, 17), fill=DIM)
frames += to_frames(img, 4)

# --------------------------------------------------------- 2. architecture
img, d = slide()
kicker(d, 'SYSTEM ARCHITECTURE')
y = 92
boxes = [
    ('nuscenes_publisher', 'replays nuScenes keyframes + ground truth', ACCENT),
    ('detector_node', 'FCOS3D inference, publishes 3D detections', GOOD),
]
d.text((56, y), 'nuScenes v1.0-mini  →  ROS 2 topics  →  FCOS3D  →  Detection3DArray',
       font=F(REG, 19), fill=DIM)
y += 48
for name, desc, col in boxes:
    d.rectangle([56, y, W - 56, y + 76], outline=col, width=2)
    d.text((76, y + 16), name, font=F(MONO, 21), fill=col)
    d.text((76, y + 46), desc, font=F(REG, 16), fill=DIM)
    y += 96
rows = [
    ('sub', '/camera/front/image_raw', 'sensor_msgs/Image'),
    ('sub', '/camera/front/camera_info', 'sensor_msgs/CameraInfo'),
    ('pub', '/perception/detections_3d', 'vision_msgs/Detection3DArray'),
    ('pub', '/perception/markers', 'visualization_msgs/MarkerArray'),
]
y += 8
for kind, topic, typ in rows:
    col = ACCENT if kind == 'sub' else GOOD
    d.text((56, y), kind, font=F(MONO, 14), fill=col)
    d.text((100, y), topic, font=F(MONO, 14), fill=FG)
    d.text((400, y), typ, font=F(MONO, 14), fill=DIM)
    y += 24
frames += to_frames(img, 7)

# ------------------------------------------------------------- 3. live demo
img, d = slide()
kicker(d, 'LIVE PIPELINE')
d.text((56, 200), 'FCOS3D running as a ROS 2 node', font=F(BOLD, 38), fill=FG)
d.text((56, 258), 'monocular RGB in  →  3D boxes out', font=F(REG, 24), fill=ACCENT)
d.text((56, 320), 'coloured = prediction     green = ground truth', font=F(REG, 19), fill=DIM)
d.text((56, 352), 'no LiDAR, no depth sensor, no temporal context', font=F(REG, 19), fill=DIM)
frames += to_frames(img, 3)

cap = cv2.VideoCapture(CLIP)
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if fr.shape[1] != W or fr.shape[0] != H:
        fr = cv2.resize(fr, (W, H))
    frames.append(fr)
    n += 1
cap.release()
print(f'  footage: {n} frames')

# ------------------------------------------------------------- 4. accuracy
img, d = slide()
kicker(d, 'DETECTION ACCURACY  ·  official nuScenes evaluation, mini_val')
d.text((56, 88), 'mAP 0.2943', font=F(BOLD, 46), fill=FG)
d.text((330, 88), 'NDS 0.3217', font=F(BOLD, 46), fill=FG)
d.text((56, 150), 'but the aggregate is misleading', font=F(BOLD, 21), fill=WARN)
d.text((56, 182), 'barrier, trailer and construction_vehicle have ZERO ground truth',
       font=F(REG, 17), fill=DIM)
d.text((56, 206), 'in this split. The evaluator scores them 0.000 and still averages',
       font=F(REG, 17), fill=DIM)
d.text((56, 230), 'over all ten classes:  2.943 / 10 = 0.2943  (exactly).',
       font=F(MONO, 16), fill=FG)
y = 280
d.text((56, y), 'mAP over the 7 classes actually present', font=F(REG, 19), fill=DIM)
d.text((620, y - 6), '0.4204', font=F(BOLD, 30), fill=GOOD)
y += 52
per = [('traffic_cone', .592), ('bus', .497), ('car', .496), ('pedestrian', .432),
       ('truck', .385), ('motorcycle', .317), ('bicycle', .224)]
for i, (name, ap) in enumerate(per):
    x = 56 + (i % 4) * 226
    yy = y + (i // 4) * 48
    d.text((x, yy), name, font=F(REG, 14), fill=DIM)
    d.rectangle([x, yy + 20, x + 190, yy + 30], fill=(30, 36, 46))
    d.rectangle([x, yy + 20, x + int(190 * ap / .8), yy + 30], fill=ACCENT)
    d.text((x + 196, yy + 16), f'{ap:.2f}', font=F(MONO, 14), fill=FG)
frames += to_frames(img, 9)

# ---------------------------------------------------------- 5. environments
img, d = slide()
kicker(d, 'REQUIREMENT 4  ·  latency & resources across computing environments')
hdr = ['', 'RTX 4090 FP32', 'RTX 4090 FP16', 'A40 FP32', 'A40 FP16']
rows = [
    ('mean latency', '72.05 ms', '74.85 ms', '144.57 ms', '134.79 ms'),
    ('p95 latency', '97.75 ms', '98.44 ms', '191.36 ms', '196.52 ms'),
    ('throughput', '13.85 FPS', '13.33 FPS', '6.91 FPS', '7.41 FPS'),
    ('GPU util', '54.7 %', '36.0 %', '53.6 %', '38.3 %'),
    ('peak GPU mem', '597 MB', '576 MB', '597 MB', '576 MB'),
    ('CPU (96 cores)', '26.5 %', '25.4 %', '10.4 %', '11.5 %'),
    ('detections/frame', '6.28', '6.28', '6.28', '6.30'),
]
cols = [56, 250, 400, 560, 710]
y = 96
for i, h in enumerate(hdr):
    d.text((cols[i], y), h, font=F(BOLD, 14), fill=ACCENT)
y += 26
d.line([56, y, W - 56, y], fill=(48, 54, 61), width=1)
y += 12
for r in rows:
    d.text((cols[0], y), r[0], font=F(REG, 15), fill=DIM)
    for i in range(1, 5):
        col = FG
        if r[0] == 'mean latency' and i == 1: col = GOOD
        d.text((cols[i], y), r[i], font=F(MONO, 15), fill=col)
    y += 30
d.text((56, y + 16), 'RTX 4090 is exactly 2.0x faster than A40  (144.57 / 72.05 = 2.007)',
       font=F(BOLD, 17), fill=GOOD)
d.text((56, y + 42), 'identical peak memory and detections/frame  →  same computation, only speed differs',
       font=F(REG, 15), fill=DIM)
frames += to_frames(img, 10)

# ------------------------------------------------------------ 6. fp16 finding
img, d = slide()
kicker(d, 'FINDING  ·  mixed precision is not a free win')
d.text((56, 100), 'FP16 helps the A40 and hurts the RTX 4090', font=F(BOLD, 30), fill=FG)
y = 175
for gpu, delta, col, note in [
    ('A40', '+6.8 % faster', GOOD, '144.57 → 134.79 ms'),
    ('RTX 4090', '−3.9 % slower', BAD, '72.05 → 74.85 ms'),
]:
    d.text((56, y), gpu, font=F(BOLD, 24), fill=FG)
    d.text((250, y), delta, font=F(BOLD, 24), fill=col)
    d.text((520, y + 4), note, font=F(MONO, 17), fill=DIM)
    y += 56
d.text((56, 310), 'GPU utilisation peaks near 54 % in FP32 on both cards.',
       font=F(REG, 19), fill=FG)
d.text((56, 340), 'The pipeline is not GPU-compute-bound, so halving arithmetic',
       font=F(REG, 19), fill=DIM)
d.text((56, 366), 'precision buys nothing on fast hardware — only autocast overhead remains.',
       font=F(REG, 19), fill=DIM)
d.text((56, 424), 'This is also why TensorRT was dropped: it optimises the part', font=F(REG, 17), fill=WARN)
d.text((56, 448), 'that is not the bottleneck.', font=F(REG, 17), fill=WARN)
frames += to_frames(img, 10)

# ---------------------------------------------------------- 7. lighting sweep
img, d = slide()
kicker(d, 'CONTROLLED LIGHTING SWEEP  ·  full evaluation at each point')
d.text((56, 88), 'Accuracy degrades nonlinearly with exposure', font=F(BOLD, 26), fill=FG)
pts = [('+20%', .3022, GOOD), ('baseline', .2943, FG), ('−20%', .2874, FG),
       ('−40%', .2706, WARN), ('−60%', .1962, BAD), ('−80%', .1202, BAD)]
x0, y0, bh = 70, 300, 150
for i, (lab, m, col) in enumerate(pts):
    x = x0 + i * 140
    h = int(bh * m / .32)
    d.rectangle([x, y0 - h, x + 96, y0], fill=col)
    d.text((x, y0 + 10), lab, font=F(BOLD, 16), fill=DIM)
    d.text((x, y0 + 32), f'{m:.3f}', font=F(MONO, 15), fill=FG)
d.text((56, y0 + 74), 'usable band down to ~0.6x brightness, then collapse  (−33 % at 0.4x, −59 % at 0.2x)',
       font=F(REG, 17), fill=DIM)
d.text((56, y0 + 106), 'class spread at dusk:   bus −95 %      motorcycle −38 %      traffic_cone −12 %',
       font=F(BOLD, 18), fill=WARN)
frames += to_frames(img, 10)

# ------------------------------------------------------------- 8. wrap
img, d = slide()
kicker(d, 'SUMMARY')
items = [
    ('Pretrained model, structure and I/O', 'FCOS3D R101-DCN + FPN, camera-only'),
    ('ROS 2 node with defined topics', 'detector_node + nuscenes_publisher'),
    ('Applied to a real-world dataset', 'nuScenes v1.0-mini, 486 images evaluated'),
    ('Accuracy, latency, CPU/GPU/memory', 'across 2 GPUs and 2 precisions'),
]
y = 100
for title, sub in items:
    d.text((56, y), '✓', font=F(BOLD, 22), fill=GOOD)
    d.text((90, y), title, font=F(BOLD, 19), fill=FG)
    d.text((90, y + 26), sub, font=F(REG, 16), fill=DIM)
    y += 66
d.text((56, y + 20), 'github.com/crgonzales/sim2real-fcos3d-ros2', font=F(MONO, 20), fill=ACCENT)
d.text((56, y + 52), 'full write-up: docs/PROJECT_REPORT.md', font=F(REG, 16), fill=DIM)
frames += to_frames(img, 7)

# ------------------------------------------------------------------ encode
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
for f in frames:
    vw.write(f)
vw.release()
print(f'wrote {OUT}: {len(frames)} frames, {len(frames)/FPS:.1f}s')
