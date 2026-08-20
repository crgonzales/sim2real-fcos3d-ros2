#!/usr/bin/env python3
"""Assemble the CMPE 249 demo video: slides + live detection footage.

Renders a ~4.5 minute self-contained walkthrough of the project. Slides are
drawn with PIL (crisper text than cv2.putText) and interleaved with footage
produced by scripts/render_demo.py.

Usage:
    python3 scripts/build_demo_video.py

Contributor: Carlos Gonzales
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 960, 540, 10
OUT = 'docs/media/cmpe249_demo.mp4'
# Footage produced by scripts/render_demo.py. Override with FOOTAGE_DIR.
FOOTAGE_DIR = os.environ.get('FOOTAGE_DIR', 'docs/media')
CLIP_MAIN = f'{FOOTAGE_DIR}/demo_footage_40s.mp4'
CLIP_ALT = f'{FOOTAGE_DIR}/demo_footage_dusk_10s.mp4'

FD = '/System/Library/Fonts/Supplemental/'
def F(n, s): return ImageFont.truetype(FD + n, s)
BOLD, REG, MONO = 'Arial Bold.ttf', 'Arial.ttf', 'Courier New Bold.ttf'

BG, FG, DIM = (14, 17, 23), (232, 237, 243), (139, 148, 158)
ACCENT, GOOD, WARN, BAD = (88, 166, 255), (63, 185, 80), (210, 153, 34), (248, 81, 73)
PANEL = (22, 27, 34)

frames = []

def slide(kicker_text=None):
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 4], fill=ACCENT)
    if kicker_text:
        d.text((56, 34), kicker_text, font=F(BOLD, 14), fill=ACCENT)
    return img, d

# Slide durations are authored at a comfortable reading pace, then scaled to
# hit the target runtime. Footage is never scaled.
SLIDE_SCALE = 0.82

def hold(img, seconds):
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    frames.extend([arr] * max(int(seconds * SLIDE_SCALE * FPS), FPS))

def check(d, x, y, size=20, color=(63, 185, 80), w=3):
    """Draw a tick. Arial has no U+2713, so text('✓') renders as tofu."""
    d.line([(x, y + size * 0.55), (x + size * 0.35, y + size * 0.85)],
           fill=color, width=w)
    d.line([(x + size * 0.35, y + size * 0.85), (x + size * 0.95, y + size * 0.15)],
           fill=color, width=w)


def bullets(d, items, y0, gap=42, size=19, color=FG, bullet_color=ACCENT):
    y = y0
    for it in items:
        d.text((58, y), '—', font=F(BOLD, size), fill=bullet_color)
        d.text((88, y), it, font=F(REG, size), fill=color)
        y += gap
    return y

def add_clip(path, max_seconds=None):
    cap = cv2.VideoCapture(path)
    n, lim = 0, (int(max_seconds * FPS) if max_seconds else 10**9)
    while n < lim:
        ok, fr = cap.read()
        if not ok:
            break
        if (fr.shape[1], fr.shape[0]) != (W, H):
            fr = cv2.resize(fr, (W, H))
        frames.append(fr)
        n += 1
    cap.release()
    print(f'  clip {os.path.basename(path)}: {n} frames ({n/FPS:.1f}s)')

# ============================================================== 1. TITLE
img, d = slide()
d.text((56, 140), 'Applying and Evaluating', font=F(REG, 38), fill=DIM)
d.text((56, 186), 'a Pretrained Monocular', font=F(BOLD, 44), fill=FG)
d.text((56, 238), '3D Object Detector in ROS 2', font=F(BOLD, 44), fill=FG)
d.text((56, 314), 'FCOS3D   ·   ROS 2 Humble   ·   nuScenes', font=F(REG, 22), fill=ACCENT)
d.text((56, 408), 'Carlos Gonzales', font=F(BOLD, 25), fill=FG)
d.text((56, 442), 'CMPE 249   ·   Option 1 (Apply & Evaluate)   ·   Focused area: Application',
       font=F(REG, 16), fill=DIM)
hold(img, 7)

# ============================================================== 2. THE TASK
img, d = slide('THE ASSIGNMENT')
d.text((56, 78), 'Deploy an existing pretrained model — no training', font=F(BOLD, 27), fill=FG)
y = bullets(d, [
    'Select a state-of-the-art pretrained model, understand its structure and I/O',
    'Wrap it in a ROS 2 node with clearly defined input/output topics',
    'Apply it to a ROS 2 simulation or a real-world dataset',
    'Evaluate accuracy, latency and CPU/GPU/memory across computing environments',
], 150, gap=48, size=18)
d.rectangle([56, y + 20, W - 56, y + 96], fill=PANEL)
d.text((76, y + 36), 'All four are implemented and independently verified.',
       font=F(BOLD, 19), fill=GOOD)
d.text((76, y + 64), 'This video walks through the model, the system, and the results.',
       font=F(REG, 17), fill=DIM)
hold(img, 11)

# ============================================================== 3. MODEL
img, d = slide('SELECTED MODEL')
d.text((56, 74), 'FCOS3D', font=F(BOLD, 46), fill=FG)
d.text((250, 92), 'Wang et al., ICCV Workshops 2021', font=F(REG, 19), fill=DIM)
d.text((56, 140), 'Fully Convolutional One-Stage Monocular 3D Object Detection',
       font=F(REG, 21), fill=ACCENT)
bullets(d, [
    'Anchor-free, single-stage, camera-only',
    'One RGB image in — full 3D boxes out',
    'No LiDAR, no depth sensor, no temporal context',
    'Pretrained on nuScenes, published by MMDetection3D',
], 196, gap=42, size=19)
d.rectangle([56, 400, W - 56, 480], fill=PANEL)
d.text((76, 416), 'Chosen because the I/O interface stays simple enough to wrap',
       font=F(REG, 17), fill=DIM)
d.text((76, 444), 'in a node, and it runs near real time on one GPU.',
       font=F(REG, 17), fill=DIM)
hold(img, 12)

# ============================================================== 4. ARCH
img, d = slide('MODEL STRUCTURE')
stages = [
    ('Backbone', 'ResNet-101  +  deformable convolutions (DCNv2) in stages 3–4', ACCENT),
    ('Neck', 'Feature Pyramid Network, 5 levels (P3–P7)', GOOD),
    ('Head', 'shared FCOSMono3DHead with per-level towers', WARN),
]
y = 92
for name, desc, col in stages:
    d.rectangle([56, y, W - 56, y + 74], outline=col, width=2)
    d.text((78, y + 14), name, font=F(BOLD, 22), fill=col)
    d.text((78, y + 44), desc, font=F(REG, 17), fill=DIM)
    y += 92
d.text((56, y + 12), 'The head regresses, at every feature location:', font=F(BOLD, 18), fill=FG)
d.text((56, y + 44),
       '2D offset to the projected 3D centre  ·  depth  ·  dimensions  ·  orientation (sin/cos)',
       font=F(REG, 16), fill=DIM)
d.text((56, y + 70), 'velocity  ·  centreness  ·  class logits  ·  attribute',
       font=F(REG, 16), fill=DIM)
hold(img, 13)

# ============================================================== 5. I/O
img, d = slide('INPUTS AND OUTPUTS  ·  the conventions that silently break things')
d.text((56, 74), 'Input', font=F(BOLD, 24), fill=ACCENT)
d.text((56, 110), '1600 × 900 RGB image  +  3×3 camera intrinsics', font=F(REG, 19), fill=FG)
d.rectangle([56, 146, W - 56, 210], fill=PANEL)
d.text((76, 160), 'bgr_to_rgb=False, caffe means [103.53, 116.28, 123.675]',
       font=F(MONO, 15), fill=WARN)
d.text((76, 184), 'the model expects BGR — feeding RGB degrades accuracy with no error',
       font=F(REG, 16), fill=DIM)
d.text((56, 234), 'Output', font=F(BOLD, 24), fill=GOOD)
d.text((56, 270), 'CameraInstance3DBoxes in the camera frame', font=F(REG, 19), fill=FG)
d.text((56, 300), '(x, y, z, x_size, y_size, z_size, yaw)', font=F(MONO, 17), fill=FG)
d.rectangle([56, 336, W - 56, 442], fill=PANEL)
d.text((76, 350), 'origin = (0.5, 1.0, 0.5)  →  BOTTOM-centre, not centre',
       font=F(MONO, 15), fill=WARN)
d.text((76, 376), 'camera frame is x-right, y-down, z-forward; yaw about y',
       font=F(MONO, 15), fill=DIM)
d.text((76, 404), 'get this wrong and every box shifts by half its height',
       font=F(REG, 16), fill=BAD)
d.text((56, 466), 'The node uses bboxes_3d.gravity_center, not the raw tensor columns.',
       font=F(REG, 17), fill=GOOD)
hold(img, 15)

# ============================================================== 6. SYSTEM
img, d = slide('SYSTEM ARCHITECTURE')
y = 96
for name, desc, col in [
    ('nuscenes_publisher', 'replays nuScenes keyframes + ground truth onto topics', ACCENT),
    ('detector_node', 'FCOS3D inference, publishes 3D detections + markers', GOOD),
]:
    d.rectangle([56, y, W - 56, y + 80], outline=col, width=2)
    d.text((78, y + 16), name, font=F(MONO, 22), fill=col)
    d.text((78, y + 48), desc, font=F(REG, 17), fill=DIM)
    y += 104
rows = [
    ('sub', '/camera/front/image_raw', 'sensor_msgs/Image', ACCENT),
    ('sub', '/camera/front/camera_info', 'sensor_msgs/CameraInfo', ACCENT),
    ('pub', '/perception/detections_3d', 'vision_msgs/Detection3DArray', GOOD),
    ('pub', '/perception/markers', 'visualization_msgs/MarkerArray', GOOD),
    ('pub', '/nuscenes/gt_markers', 'visualization_msgs/MarkerArray', WARN),
]
y += 6
for kind, topic, typ, col in rows:
    d.text((58, y), kind, font=F(MONO, 14), fill=col)
    d.text((102, y), topic, font=F(MONO, 14), fill=FG)
    d.text((406, y), typ, font=F(MONO, 14), fill=DIM)
    y += 25
hold(img, 13)

# ============================================================== 7. DESIGN
img, d = slide('TWO DESIGN DECISIONS')
d.text((56, 76), 'Ground truth is published in the camera frame', font=F(BOLD, 23), fill=GOOD)
bullets(d, [
    'the same frame FCOS3D predicts in, so GT and predictions overlay directly',
    'a silent frame mismatch becomes something you can SEE, not debug blind',
], 120, gap=34, size=17, bullet_color=GOOD)
d.text((56, 226), 'Sensor QoS is best-effort, depth 1', font=F(BOLD, 23), fill=ACCENT)
bullets(d, [
    'a detector slower than the publisher drops frames',
    'rather than accumulating unbounded latency',
    'correct for perception: a stale detection is worse than none',
], 270, gap=34, size=17)
d.rectangle([56, 408, W - 56, 484], fill=PANEL)
d.text((76, 424), 'This mattered: a wrong-camera intrinsics bug early on produced',
       font=F(REG, 17), fill=DIM)
d.text((76, 452), 'plausible-looking boxes and no error at all.', font=F(REG, 17), fill=WARN)
hold(img, 13)

# ============================================================== 8. LIVE
img, d = slide('LIVE PIPELINE')
d.text((56, 178), 'FCOS3D running as a ROS 2 node', font=F(BOLD, 40), fill=FG)
d.text((56, 240), 'monocular RGB in   →   3D boxes out', font=F(REG, 26), fill=ACCENT)
d.text((56, 312), 'coloured = prediction          green = ground truth', font=F(REG, 20), fill=DIM)
d.text((56, 348), 'HUD shows per-frame latency, running mean and FPS', font=F(REG, 20), fill=DIM)
hold(img, 5)
add_clip(CLIP_MAIN, max_seconds=40)

# ============================================================== 9. ACCURACY
img, d = slide('DETECTION ACCURACY  ·  official nuScenes evaluation, mini_val')
d.text((56, 84), 'mAP 0.2943', font=F(BOLD, 44), fill=FG)
d.text((340, 84), 'NDS 0.3217', font=F(BOLD, 44), fill=FG)
d.text((56, 146), '81 keyframes · 2 scenes · 486 camera images', font=F(REG, 18), fill=DIM)
tp = [('mATE  translation', '0.7782 m'), ('mASE  scale', '0.4671'),
      ('mAOE  orientation', '0.7229 rad'), ('mAVE  velocity', '1.2499 m/s'),
      ('mAAE  attribute', '0.2865')]
y = 200
for k, v in tp:
    d.text((58, y), k, font=F(REG, 18), fill=DIM)
    d.text((330, y), v, font=F(MONO, 18), fill=FG)
    y += 34
d.rectangle([56, y + 14, W - 56, y + 92], fill=PANEL)
d.text((76, y + 30), 'mATE dominates the error budget — monocular depth is', font=F(REG, 17), fill=DIM)
d.text((76, y + 58), 'the fundamental limitation of a single-image method.', font=F(REG, 17), fill=DIM)
hold(img, 13)

# ============================================================== 10. CAVEAT
img, d = slide('BUT THE HEADLINE NUMBER IS MISLEADING')
d.text((56, 76), 'Three classes have ZERO ground truth in this split', font=F(BOLD, 25), fill=WARN)
bullets(d, [
    'barrier, trailer and construction_vehicle never occur in scenes 0103 / 0916',
    'nuScenes scores an absent class AP 0.000 and still averages over all ten',
], 128, gap=36, size=18, bullet_color=WARN)
d.rectangle([56, 214, W - 56, 268], fill=PANEL)
d.text((76, 232), '2.943 / 10  =  0.2943      — exactly the reported mAP',
       font=F(MONO, 20), fill=FG)
y = 296
for label, val, col in [('all 10 classes (as reported)', '0.2943', DIM),
                        ('7 classes actually present', '0.4204', GOOD),
                        ('published, full val (all present)', '0.299', DIM)]:
    d.text((58, y), label, font=F(REG, 19), fill=DIM)
    d.text((560, y - 4), val, font=F(BOLD, 26), fill=col)
    y += 46
d.text((56, y + 16), 'So the apparent agreement with the published 0.299 is a coincidence.',
       font=F(REG, 17), fill=WARN)
d.text((56, y + 42), 'What validates the pipeline is the per-class results being sensible.',
       font=F(REG, 17), fill=DIM)
hold(img, 17)

# ============================================================== 11. PER-CLASS
img, d = slide('PER-CLASS AVERAGE PRECISION')
per = [('traffic_cone', .592), ('bus', .497), ('car', .496), ('pedestrian', .432),
       ('truck', .385), ('motorcycle', .317), ('bicycle', .224)]
y = 110
for name, ap in per:
    d.text((58, y), name, font=F(REG, 18), fill=DIM)
    d.rectangle([250, y + 2, 250 + 480, y + 22], fill=(30, 36, 46))
    d.rectangle([250, y + 2, 250 + int(480 * ap / .65), y + 22], fill=ACCENT)
    d.text((748, y), f'{ap:.3f}', font=F(MONO, 18), fill=FG)
    y += 44
d.text((56, y + 20), 'Strongest on cones, cars and buses; weakest on bicycles.',
       font=F(REG, 18), fill=DIM)
d.text((56, y + 50), 'barrier / trailer / construction_vehicle omitted — no ground truth.',
       font=F(REG, 17), fill=WARN)
hold(img, 13)

# ============================================================== 12. ENV TABLE
img, d = slide('REQUIREMENT 4  ·  latency and resources across computing environments')
hdr = ['', 'RTX 4090 FP32', 'RTX 4090 FP16', 'A40 FP32', 'A40 FP16']
rows = [
    ('mean latency', '72.05 ms', '74.85 ms', '144.57 ms', '134.79 ms'),
    ('p95 latency', '97.75 ms', '98.44 ms', '191.36 ms', '196.52 ms'),
    ('throughput', '13.85 FPS', '13.33 FPS', '6.91 FPS', '7.41 FPS'),
    ('GPU utilisation', '54.7 %', '36.0 %', '53.6 %', '38.3 %'),
    ('GPU memory', '2102 MB', '1781 MB', '2051 MB', '1732 MB'),
    ('torch peak alloc', '597 MB', '576 MB', '597 MB', '576 MB'),
    ('process RSS *', '1905 MB', '1999 MB', '1915 MB', '2016 MB'),
    ('CPU (96 cores)', '26.5 %', '25.4 %', '10.4 %', '11.5 %'),
    ('detections/frame', '6.28', '6.28', '6.28', '6.30'),
]
cols = [58, 250, 400, 560, 715]
y = 92
for i, h in enumerate(hdr):
    d.text((cols[i], y), h, font=F(BOLD, 14), fill=ACCENT)
y += 24
d.line([56, y, W - 56, y], fill=(48, 54, 61), width=1)
y += 10
for r in rows:
    d.text((cols[0], y), r[0], font=F(REG, 15), fill=DIM)
    for i in range(1, 5):
        d.text((cols[i], y), r[i], font=F(MONO, 15),
               fill=GOOD if (r[0] == 'mean latency' and i == 1) else FG)
    y += 28
d.text((56, y + 12), 'identical peak memory and detections/frame → both machines run the same computation',
       font=F(REG, 15), fill=DIM)
d.text((56, y + 38), '* RSS is inflated ~432 MB on every column: the profiler cached the benchmark corpus.',
       font=F(REG, 14), fill=WARN)
d.text((56, y + 60), '  Found in code review and fixed. Re-measured on the 4090: 1477 MB. Latency and GPU figures unaffected.',
       font=F(REG, 14), fill=WARN)
hold(img, 16)

# ============================================================== 13. 2x
img, d = slide('FINDING 1')
d.text((56, 96), 'RTX 4090 is 2.0× faster than A40', font=F(BOLD, 34), fill=FG)
d.text((56, 152), '144.57 / 72.05  =  2.007', font=F(MONO, 24), fill=ACCENT)
y = 214
for k, v, col in [('RTX 4090', '$0.053 per FPS', GOOD), ('A40', '$0.064 per FPS', DIM)]:
    d.text((58, y), k, font=F(BOLD, 22), fill=FG)
    d.text((300, y), v, font=F(MONO, 22), fill=col)
    y += 46
d.text((56, 320), 'The 4090 is better value despite costing 68% more per hour.',
       font=F(REG, 19), fill=DIM)
d.rectangle([56, 366, W - 56, 476], fill=PANEL)
d.text((76, 382), 'Caveat, stated in the report:', font=F(BOLD, 17), fill=WARN)
d.text((76, 410), 'the host CPU was not controlled, and ~46% of wall time is', font=F(REG, 16), fill=DIM)
d.text((76, 434), 'host-side. This is an end-to-end pod measurement, not a', font=F(REG, 16), fill=DIM)
d.text((76, 458), 'pure GPU benchmark.', font=F(REG, 16), fill=DIM)
hold(img, 15)

# ============================================================== 14. FP16
img, d = slide('FINDING 2  ·  mixed precision is not a free win')
d.text((56, 92), 'FP16 helps the A40 and hurts the RTX 4090', font=F(BOLD, 30), fill=FG)
y = 168
for gpu, delta, col, note in [('A40', '+6.8 % faster', GOOD, '144.57 → 134.79 ms'),
                              ('RTX 4090', '−3.9 % slower', BAD, '72.05 → 74.85 ms')]:
    d.text((58, y), gpu, font=F(BOLD, 24), fill=FG)
    d.text((250, y), delta, font=F(BOLD, 24), fill=col)
    d.text((520, y + 4), note, font=F(MONO, 17), fill=DIM)
    y += 54
d.text((56, 300), 'GPU utilisation peaks near 54 % in FP32 on both cards.', font=F(REG, 19), fill=FG)
d.text((56, 332), 'The pipeline is not GPU-compute-bound, so halving arithmetic', font=F(REG, 19), fill=DIM)
d.text((56, 358), 'precision buys nothing on fast hardware — only autocast overhead remains.',
       font=F(REG, 19), fill=DIM)
d.rectangle([56, 400, W - 56, 484], fill=PANEL)
d.text((76, 416), 'This is also why TensorRT was dropped rather than deferred:', font=F(REG, 17), fill=WARN)
d.text((76, 444), 'it optimises GPU kernel execution — the part that is not', font=F(REG, 17), fill=WARN)
d.text((76, 468), 'the bottleneck.', font=F(REG, 17), fill=WARN)
hold(img, 16)

# ============================================================== 15. SWEEP INTRO
img, d = slide('CONTROLLED LIGHTING SWEEP')
d.text((56, 88), 'The proposal asked: which classes suffer under', font=F(BOLD, 26), fill=FG)
d.text((56, 124), 'changed appearance, and by how much?', font=F(BOLD, 26), fill=FG)
bullets(d, [
    'Isaac Sim was scoped out — the assignment requires simulation OR a dataset',
    'Instead: perturb real nuScenes images with a deterministic gain / gamma',
    'Run the FULL official evaluation at every sweep point',
    'One variable at a time; scene, geometry, ground truth and model held fixed',
], 190, gap=42, size=18)
d.rectangle([56, 384, W - 56, 486], fill=PANEL)
d.text((76, 400), 'A rendering-gap proxy, not a simulator substitute — it changes', font=F(REG, 17), fill=DIM)
d.text((76, 426), 'photometric response only. The baseline point reproduces the', font=F(REG, 17), fill=DIM)
d.text((76, 452), 'unperturbed run to four decimals, so the transform is a verified no-op.',
       font=F(REG, 17), fill=GOOD)
hold(img, 16)

# ============================================================== 16. SWEEP CHART
img, d = slide('ACCURACY vs EXPOSURE')
pts = [('+20%', .3022, GOOD), ('baseline', .2943, FG), ('−20%', .2874, FG),
       ('−40%', .2706, WARN), ('−60%', .1962, BAD), ('−80%', .1202, BAD)]
x0, y0, bh = 74, 330, 190
for i, (lab, m, col) in enumerate(pts):
    x = x0 + i * 138
    hgt = int(bh * m / .32)
    d.rectangle([x, y0 - hgt, x + 96, y0], fill=col)
    d.text((x, y0 + 12), lab, font=F(BOLD, 16), fill=DIM)
    d.text((x, y0 + 34), f'{m:.3f}', font=F(MONO, 15), fill=FG)
d.text((56, 92), 'Degradation is strongly nonlinear', font=F(BOLD, 26), fill=FG)
d.text((56, 400), 'usable band down to ~0.6× brightness, then collapse',
       font=F(REG, 19), fill=FG)
d.text((56, 432), '−33 % mAP at 0.4×      −59 % mAP at 0.2×', font=F(MONO, 18), fill=BAD)
d.text((56, 470), 'a renderer only has to land inside that band to give trustworthy numbers',
       font=F(REG, 17), fill=DIM)
hold(img, 16)

# ============================================================== 17. SWEEP CLASSES
img, d = slide('WHICH CLASSES SUFFER  ·  at 0.4× brightness')
rows = [('bus', .647, .033, '−95 %', BAD), ('motorcycle', .452, .281, '−38 %', BAD),
        ('car', .680, .494, '−27 %', WARN), ('bicycle', .332, .244, '−27 %', WARN),
        ('truck', .522, .434, '−17 %', WARN), ('pedestrian', .578, .477, '−17 %', WARN),
        ('traffic_cone', .740, .648, '−12 %', GOOD)]
y = 104
d.text((58, y), 'class', font=F(BOLD, 14), fill=ACCENT)
d.text((250, y), 'baseline', font=F(BOLD, 14), fill=ACCENT)
d.text((380, y), 'dusk', font=F(BOLD, 14), fill=ACCENT)
d.text((500, y), 'change', font=F(BOLD, 14), fill=ACCENT)
y += 30
for name, b, a, delta, col in rows:
    d.text((58, y), name, font=F(REG, 18), fill=FG)
    d.text((250, y), f'{b:.3f}', font=F(MONO, 17), fill=DIM)
    d.text((380, y), f'{a:.3f}', font=F(MONO, 17), fill=FG)
    d.text((500, y), delta, font=F(BOLD, 18), fill=col)
    y += 40
d.rectangle([56, y + 14, W - 56, y + 96], fill=PANEL)
d.text((76, y + 30), 'Cones and pedestrians are recognised largely by silhouette,', font=F(REG, 17), fill=DIM)
d.text((76, y + 56), 'which survives loss of contrast. Buses are large low-texture', font=F(REG, 17), fill=DIM)
d.text((76, y + 78), 'surfaces whose detail washes out. (Hypothesis, not established.)',
       font=F(REG, 16), fill=WARN)
hold(img, 17)

# ============================================================== 18. SECOND CLIP
img, d = slide('DETECTOR OUTPUT  ·  predictions only')
d.text((56, 200), 'Same pipeline, ground truth hidden', font=F(BOLD, 32), fill=FG)
d.text((56, 258), 'so the raw detector output is easier to read', font=F(REG, 22), fill=DIM)
hold(img, 4)
add_clip(CLIP_ALT, max_seconds=10)

# ============================================================== 19. ENGINEERING
img, d = slide('ENGINEERING REALITY  ·  where the time actually went')
d.text((56, 74), 'Roughly two thirds of the effort was environment and', font=F(BOLD, 22), fill=FG)
d.text((56, 104), 'correctness work, not writing the node.', font=F(BOLD, 22), fill=FG)
items = [
    ('numpy 2.x cascade', 'mmcv is built against the numpy 1.x ABI'),
    ('mixed installs', 'pip overwrites compiled packages without removing the old tree'),
    ('silent intrinsics bug', 'CAM_FRONT_LEFT image + CAM_FRONT intrinsics → 19 vs 200 detections'),
    ('blinker / open3d', 'a distutils package pip cannot uninstall'),
    ('ros2 run found nothing', 'missing setup.cfg → scripts installed to bin/ not lib/'),
    ('hardcoded trainval', 'NuScenesDataset.METAINFO overrode the mini split'),
]
y = 156
for k, v in items:
    d.text((58, y), k, font=F(BOLD, 16), fill=WARN)
    d.text((300, y), v, font=F(REG, 15), fill=DIM)
    y += 36
d.rectangle([56, y + 10, W - 56, y + 86], fill=PANEL)
d.text((76, y + 26), 'All captured in scripts/setup_pod.sh — verified by reproducing', font=F(REG, 17), fill=GOOD)
d.text((76, y + 54), 'the whole environment from scratch on a second machine.', font=F(REG, 17), fill=GOOD)
hold(img, 17)

# ============================================================== 20. REVIEW
img, d = slide('CODE REVIEW')
d.text((56, 74), 'Independently reviewed to convergence', font=F(BOLD, 26), fill=FG)
d.text((56, 112), 'four turns, nine findings, all addressed', font=F(REG, 18), fill=DIM)
d.text((56, 154), 'What the loop caught', font=F(BOLD, 20), fill=WARN)
for i, line in enumerate([
        'default launch failed after the documented bootstrap',
        "the node's FP16 path omitted its own compatibility patch",
        'every ground-truth marker had permuted dimensions',
        'reported memory included the benchmark corpus, not the node',
        'a fix for that regressed unreadable-frame handling']):
    d.text((70, 186 + i * 26), '·', font=F(BOLD, 17), fill=WARN)
    d.text((86, 186 + i * 26), line, font=F(REG, 16), fill=DIM)
d.text((56, 330), 'Two of the nine came from fixes introducing new problems —', font=F(REG, 16), fill=DIM)
d.text((56, 354), 'the iterative loop earned its cost.', font=F(REG, 16), fill=DIM)
check(d, 56, 396, size=22)
d.text((92, 396), 'APPROVED', font=F(BOLD, 26), fill=GOOD)
d.text((250, 402), 'all six checklist sections pass', font=F(REG, 18), fill=DIM)
d.text((56, 448), '27 test cases guard the conventions that fail silently —', font=F(REG, 16), fill=DIM)
d.text((56, 472), 'geometry, channel order, camera pairing, device selection.', font=F(REG, 16), fill=DIM)
hold(img, 18)

# ============================================================== 21. SUMMARY
img, d = slide('SUMMARY')
items = [
    ('Pretrained model, structure and I/O', 'FCOS3D R101-DCN + FPN, camera-only'),
    ('ROS 2 node with defined topics', 'detector_node + nuscenes_publisher, verified live'),
    ('Applied to a real-world dataset', 'nuScenes v1.0-mini, 486 images evaluated'),
    ('Accuracy, latency, CPU/GPU/memory', 'across 2 GPUs and 2 precisions'),
]
y = 92
for title, sub in items:
    check(d, 58, y + 2, size=20)
    d.text((94, y), title, font=F(BOLD, 20), fill=FG)
    d.text((94, y + 28), sub, font=F(REG, 17), fill=DIM)
    y += 72
d.rectangle([56, y + 6, W - 56, y + 74], fill=PANEL)
d.text((76, y + 22), 'Dropped and stated in the report: Isaac Sim, BEVFormer, TensorRT',
       font=F(REG, 17), fill=WARN)
d.text((76, y + 48), 'each with the reasoning, rather than passed over in silence.',
       font=F(REG, 16), fill=DIM)
d.text((56, y + 100), 'github.com/crgonzales/sim2real-fcos3d-ros2', font=F(MONO, 20), fill=ACCENT)
hold(img, 14)

# ============================================================== ENCODE
# avc1 (H.264) rather than mp4v: roughly 4x smaller and accepted by browsers
# and upload tools that reject the MPEG-4 Part 2 stream mp4v produces.
# Falls back if this OpenCV build has no H.264 encoder.
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'avc1'), FPS, (W, H))
if not vw.isOpened():
    print('  avc1 unavailable, falling back to mp4v')
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
for f in frames:
    vw.write(f)
vw.release()
secs = len(frames) / FPS
print(f'wrote {OUT}: {len(frames)} frames, {secs:.1f}s ({int(secs//60)}:{int(secs%60):02d})')
