"""
One-shot tracking test:
  1. reads the video
  2. runs connectivity tracking (captures the trajectories)
  3. saves a trajectory plot
  4. computes the MSD table from those SAME trajectories and writes a CSV

Outputs (Trajectories.png and the MSD csv) are saved in the SAME folder as
the input video, named after the video so multiple files don't collide.

MSD matches the lab's MATLAB MSD_RPS convention:
  - overlapping lags (all start times)
  - drift-subtracted:  <dx^2> - <dx>^2
  - NaN-aware means
  - stop once n < NMIN
"""

import os
import csv
import numpy as np

from core import BarcodeConfig, InputConfig
from analysis.tracking import analyze_tracking, plot_tracks, build_binary_stack
from utils.reader import read_file

# ----------------------------------------------------------------------
# SETTINGS  (edit these)
# ----------------------------------------------------------------------
FILE = r"H:\Hard drive\Nance's\PLE\100nm\others\3%\100nm\May 9\PLE18032-100x-40fps-1_MMStack_Pos0.ome.tif"

PIXEL_SIZE = 0.11        # micrometres per pixel
FRAME_RATE = 40.0        # frames per second
NMIN       = 10          # stop extending lags once fewer than this many pairs

# tracking parameters -- match whatever gave you the clean result
THRESHOLD_OFFSET   = 0.7
MIN_TRACK_LENGTH   = 1
MIN_COMPONENT_SIZE = 50
TIME_GAP           = 10
SPATIAL_DILATION   = 0
MIN_PARTICLE_SIZE = 10    # <-- your particle's rough pixel size

# ----------------------------------------------------------------------
# output locations: same folder as the input, named after the video
# ----------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(FILE)
BASE_NAME  = os.path.splitext(os.path.basename(FILE))[0]
PLOT_PATH  = os.path.join(OUTPUT_DIR, BASE_NAME + "_Trajectories.png")
OUT_CSV    = os.path.join(OUTPUT_DIR, BASE_NAME + "_msd_x.csv")

# ----------------------------------------------------------------------
# 1. config
# ----------------------------------------------------------------------
config = BarcodeConfig()
config.reader.verbose = True
config.reader.um_pixel_ratio = PIXEL_SIZE
config.reader.exposure_time = 1.0 / FRAME_RATE
config.modules.image_binarization = True
config.modules.tracking = True

config.tracking_parameters.gate_mode = "report"
config.tracking_parameters.threshold_offset = THRESHOLD_OFFSET
config.tracking_parameters.min_track_length = MIN_TRACK_LENGTH
config.tracking_parameters.min_component_size = MIN_COMPONENT_SIZE
config.tracking_parameters.time_gap = TIME_GAP
config.tracking_parameters.spatial_dilation = SPATIAL_DILATION
config.tracking_parameters.min_particle_size = MIN_PARTICLE_SIZE
config.tracking_parameters.bin_factor = 1          # REQUIRED for weighting
config.tracking_parameters.weighted_centroid = True

# ----------------------------------------------------------------------
# 2. read video + run tracking (capture the trajectories)
# ----------------------------------------------------------------------
print("File exists:", os.path.isfile(FILE))

in_config = InputConfig()
counts = [1, 1]
file = read_file(FILE, counts, config, in_config, config.reader.accept_dim_images)

if file is None:
    print("File was not read (it may have been flagged too dim). "
          "Set config.reader.accept_dim_images = True to force it.")
    raise SystemExit

video = file[:, :, :, 0]

tracks = analyze_tracking(video, ".", config.tracking_parameters,
                          config.reader, config.writer)

if not tracks:
    print("No trajectories found. (Check the feasibility gate / settings.)")
    raise SystemExit

print(f"Captured {len(tracks)} trajectories.")

# ----------------------------------------------------------------------
# 3. plot (same tracks) -> next to the input video
# ----------------------------------------------------------------------
stack = build_binary_stack(video, THRESHOLD_OFFSET,
                           config.tracking_parameters.bin_factor,
                           invert=config.tracking_parameters.invert_binarization)
try:
    plot_tracks(tracks, stack.shape, PLOT_PATH)
    print("Saved", PLOT_PATH)
except Exception as e:
    print("Could not plot trajectories:", e)

# ----------------------------------------------------------------------
# 4. MSD from the SAME trajectories
# ----------------------------------------------------------------------
num_frames = video.shape[0]
particle_ids = sorted(tracks.keys())

# pivot ragged tracks -> NaN-padded (frames x particles), MATLAB xcoords layout
xcoords = np.full((num_frames, len(particle_ids)), np.nan)
for col, pid in enumerate(particle_ids):
    for (t, cx, cy) in tracks[pid]:
        xcoords[t, col] = cx

rows = num_frames
deltat, mean_dx, msd_x, n_list = [], [], [], []

for i in range(1, rows):
    dx = xcoords[i:rows, :] - xcoords[0:rows - i, :]   # overlapping lag i
    dx = dx[np.isfinite(dx)]
    n = dx.size
    if n == 0:
        break
    avg_x = np.nanmean(dx)
    avg_x2 = np.nanmean(dx ** 2)
    deltat.append(i)
    mean_dx.append(avg_x * PIXEL_SIZE)
    msd_x.append((avg_x2 - avg_x ** 2) * PIXEL_SIZE ** 2)
    n_list.append(n)
    if i % 10 == 0:
        print(f"Time Step: {i}")
    if n < NMIN:
        break

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_lag_s", "mean_dx_um", "MSDx_um2", "n"])
    for dt, mdx, m, nn in zip(deltat, mean_dx, msd_x, n_list):
        w.writerow([dt / FRAME_RATE, mdx, m, nn])

print("Saved", OUT_CSV, f"({len(deltat)} lags)")
print("Done.")