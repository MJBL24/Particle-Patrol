"""
Connectivity-based particle tracking.

Instead of detect-then-link (e.g. trackpy), this stacks the binarized frames
into an (t, y, x) volume and runs a single 3D connected-component labeling.
Each labeled 3D component that extends through time IS a trajectory, because a
particle whose blob overlaps itself between consecutive frames becomes one
connected object automatically.

This works when particles move less than ~their own size per frame (high
temporal overlap). For Brownian motion that depends on the video, so we measure
the overlap fraction per video and either gate on it or just report it.
"""

from typing import List, Tuple, Dict, Optional

import os

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops_table

from utils import vprint
from utils.binarization import invert_frame, binarize

# 1. Build the binary (t, y, x) volume from the raw video

def build_binary_stack(
    video: np.ndarray,
    threshold_offset: float,
    binning_factor: int,
    invert: bool = False,
    min_particle_size: int = 0,
    binarize_min_size: int = 1,
) -> np.ndarray:
    """
    Binarize every frame and stack into a boolean volume (t, y, x).

    min_particle_size: drop any island smaller than this many pixels IN A
        SINGLE FRAME before stacking. This is the intuitive "my particle is
        about N pixels; anything smaller is noise" filter, applied per-frame
        (like BARCODE's minimum_island_size) so noise never becomes a track.
    """
    frames = []
    for frame in video:
        b = binarize(frame, threshold_offset, binning_factor,
                     min_size=binarize_min_size)
        if invert:
            b = invert_frame(b)
        b = np.asarray(b, dtype=bool)

        if min_particle_size > 0:
            # label islands in this frame and remove those below the size floor
            lbl, n = ndimage.label(b)
            if n > 0:
                sizes = np.bincount(lbl.ravel())
                too_small = sizes < min_particle_size
                too_small[0] = False  # never remove background
                b = b & ~too_small[lbl]

        frames.append(b)
    return np.stack(frames, axis=0)

# 2. Feasibility gate: how much do blobs overlap frame-to-frame?

def temporal_overlap_fraction(stack: np.ndarray) -> float:
    """
    Mean fraction of foreground pixels in frame t that are still foreground
    in frame t+1. ~1.0 => particles barely move (great for this method).
    ~0.0 => particles jump far each frame (bad candidate).
    """
    fractions = []
    for a, b in zip(stack[:-1], stack[1:]):
        fg = a.sum()
        if fg == 0:
            continue
        overlap = np.logical_and(a, b).sum()
        fractions.append(overlap / fg)
    if not fractions:
        return 0.0
    return float(np.mean(fractions))

# 3. Link via 3D connected components
def link_by_connectivity(
    stack: np.ndarray,
    connectivity: int = 3,
    time_gap: int = 0,
    min_track_length: int = 2,
    min_component_size: int = 0,
    spatial_dilation: int = 0,
    intensity_stack: np.ndarray = None,
    weighted: bool = False,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Label the (t, y, x) volume and turn each 3D component into a trajectory.

    time_gap: dilate along the time axis by this many frames before labeling,
              to bridge short disappearances (widens the Brownian regime).
    spatial_dilation: grow each blob by this many pixels in x/y before labeling,
              so blobs can still "touch" across a larger jump between frames
              (widens the allowed step size, staying inside the connectivity
              method). Centroids are still read from the ORIGINAL blobs.
    min_component_size: drop whole 3D components smaller than this many voxels
              (kills noise speckle before it becomes fake tracks).
    intensity_stack: the original grayscale (t, y, x) frames. Required if
              weighted=True, so centroids can be brightness-weighted.
    weighted: if True, use the intensity-weighted centroid (sub-pixel, sits on
              the particle's bright peak) instead of the plain binary centroid
              (geometric centre of the blob shape).
    Returns: {track_id: [(t, cx, cy), ...]} sorted by t.
    """
    vol = stack
    if time_gap > 0:
        se = np.zeros((time_gap + 1, 1, 1), dtype=bool)
        se[:, 0, 0] = True
        vol = ndimage.binary_closing(vol, structure=se)

    if spatial_dilation > 0:
        # structuring element that grows blobs in x/y only, leaving t untouched.
        # a (1, k, k) box means: same frame, expand within the frame.
        k = 2 * spatial_dilation + 1
        se_xy = np.ones((1, k, k), dtype=bool)
        vol = ndimage.binary_dilation(vol, structure=se_xy)

    struct = ndimage.generate_binary_structure(3, connectivity)
    labels, n = ndimage.label(vol, structure=struct)
    if n == 0:
        return {}

    # Drop tiny 3D components.
    labels = labels * stack

    if min_component_size > 0:
        comp_sizes = np.bincount(labels.ravel())
        too_small = comp_sizes < min_component_size
        too_small[0] = False
        labels[too_small[labels]] = 0

    # IMPORTANT: labels were computed on the (possibly dilated) volume so that
    # gaps get bridged. But we must read centroids only where a particle
    # ACTUALLY existed. the original binary stack. Mask the labels back
    # to real foreground so the dilated "filler" pixels don't become fake
    # detections.
    

    # For each frame, get centroids of each label present in that frame.
    use_weighted = weighted and intensity_stack is not None
    tracks: Dict[int, List[Tuple[int, float, float]]] = {}
    for t in range(labels.shape[0]):
        frame_labels = labels[t]
        present = np.unique(frame_labels)
        present = present[present != 0]
        if present.size == 0:
            continue
        if use_weighted:
            # brightness-weighted centroid: sits on the particle's bright peak,
            # using the original grayscale frame as the intensity image.
            props = regionprops_table(
                frame_labels,
                intensity_image=intensity_stack[t],
                properties=["label", "weighted_centroid"],
            )
            cy_key, cx_key = "weighted_centroid-0", "weighted_centroid-1"
        else:
            props = regionprops_table(
                frame_labels, properties=["label", "centroid"]
            )
            cy_key, cx_key = "centroid-0", "centroid-1"
        for lbl, cy, cx in zip(props["label"], props[cy_key], props[cx_key]):
            tracks.setdefault(int(lbl), []).append((t, float(cx), float(cy)))

    # Drop tracks that never persist across time
    tracks = {
        k: sorted(v, key=lambda p: p[0])
        for k, v in tracks.items()
        if len(v) >= min_track_length
    }
    return tracks

# 4. Top-level entry point (mirrors analyze_binarization signature style)

def analyze_tracking(
    video: np.ndarray,
    name: str,
    track_config,
    in_config,
    out_config,
) -> Optional[Dict[int, List[Tuple[int, float, float]]]]:
    """
    Run connectivity-based tracking on one channel video.

    Behavior controlled by track_config:
      overlap_threshold : float  (0-1) minimum overlap to attempt linking
      gate_mode         : "gate" | "report"
      connectivity      : int    (1-3) 3D connectivity for labeling
      time_gap          : int     frames to bridge along time
      min_track_length  : int     drop tracks shorter than this
    """
    vprint("Beginning Connectivity-Based Tracking")

    stack = build_binary_stack(
        video,
        track_config.threshold_offset,
        track_config.bin_factor,
        invert=track_config.invert_binarization,
        min_particle_size=getattr(track_config, "min_particle_size", 0),
    )

    overlap = temporal_overlap_fraction(stack)
    vprint(f"Temporal overlap fraction: {overlap:.3f}")

    # REPORT MODE: measure only, never gate. Use this to find your threshold.
    if track_config.gate_mode == "report":
        if overlap >= track_config.overlap_threshold:
            vprint("This video is a GOOD candidate for connectivity tracking.")
        else:
            vprint("This video is a POOR candidate (particles move too far).")

    # GATE MODE: refuse to link when overlap is too low.
    elif overlap < track_config.overlap_threshold:
        vprint(
            "Low temporal overlap - not a good candidate for "
            "connectivity-based tracking (particles move too far between "
            "frames). Skipping linking."
        )
        return None

    # Intensity-weighted centroids need the grayscale frames to line up with
    # the labelled (binary) grid. If binning is on (bin_factor > 1) the binary
    # stack is a different size than the raw video, so weighting is only applied
    # when bin_factor == 1. Otherwise it silently falls back to the binary
    # centroid (and tells you why).
    want_weighted = getattr(track_config, "weighted_centroid", False)
    intensity_stack = None
    if want_weighted:
        if track_config.bin_factor == 1 and video.shape == stack.shape:
            intensity_stack = video
        else:
            vprint("weighted_centroid requested but bin_factor != 1 "
                   "(or shapes differ); using binary centroid instead. "
                   "Set bin_factor = 1 to enable weighting.")
            want_weighted = False

    tracks = link_by_connectivity(
        stack,
        connectivity=track_config.connectivity,
        time_gap=track_config.time_gap,
        min_track_length=track_config.min_track_length,
        min_component_size=getattr(track_config, "min_component_size", 0),
        spatial_dilation=getattr(track_config, "spatial_dilation", 0),
        intensity_stack=intensity_stack,
        weighted=want_weighted,
    )
    vprint(f"Found {len(tracks)} trajectories.")

    os.makedirs(name, exist_ok=True)
    # Save a picture of the trajectories so you can eyeball them.
    try:
        out_path = os.path.join(name, "Trajectories.png")
        plot_tracks(tracks, stack.shape, out_path)
        print(f"Saved trajectory plot to {out_path}")
    except Exception as e:
        print(f"Could not plot trajectories: {e}")

    # Save the MSD table (matches the lab's MATLAB MSD_RPS convention).
    try:
        csv_path = os.path.join(name, "MSD.csv")
        pixel_size = getattr(in_config, "um_pixel_ratio", 1.0) * track_config.bin_factor
        frame_interval = getattr(in_config, "exposure_time", 1.0)
        print(f"[tracking] pixel_size={pixel_size} frame_interval={frame_interval}", flush=True)
        n_frames = stack.shape[0]
        write_msd_csv(tracks, n_frames, pixel_size, frame_interval,
                      csv_path, nmin=getattr(track_config, "msd_nmin", 10))
        vprint(f"Saved MSD table to {csv_path}")
    except Exception as e:
        vprint(f"Could not compute MSD: {e}")

    return tracks

# 4b. Mean squared displacement

def compute_msd(
    tracks: Dict[int, List[Tuple[int, float, float]]],
    n_frames: int,
    pixel_size: float = 1.0,
    frame_interval: float = 1.0,
    nmin: int = 10,
):
    """
    Overlapping-lag MSD with drift subtraction, matching the MATLAB MSD_RPS
    convention:  MSD_x(tau) = ( <dx^2> - <dx>^2 ) * pixel_size^2

    Trajectories are pivoted into a NaN-padded (frames x particles) matrix so
    every particle contributes every valid (t, t+tau) pair.

    Returns lists: (t_lag_s, mean_dx_um, msd_x_um2, msd_y_um2, msd_2d_um2, n)
    """
    if not tracks:
        return [], [], [], [], [], []

    ids = sorted(tracks.keys())
    xs = np.full((n_frames, len(ids)), np.nan)
    ys = np.full((n_frames, len(ids)), np.nan)
    for col, pid in enumerate(ids):
        for (t, cx, cy) in tracks[pid]:
            if 0 <= t < n_frames:
                xs[t, col] = cx
                ys[t, col] = cy

    t_lag, mean_dx, msd_x, msd_y, msd_2d, n_list = [], [], [], [], [], []

    for i in range(1, n_frames):
        dx = xs[i:n_frames, :] - xs[0:n_frames - i, :]
        dy = ys[i:n_frames, :] - ys[0:n_frames - i, :]
        finite = np.isfinite(dx) & np.isfinite(dy)
        dx = dx[finite]
        dy = dy[finite]
        n = dx.size
        if n < max(nmin, 1):
            break

        ax_, ay_ = np.nanmean(dx), np.nanmean(dy)
        ax2, ay2 = np.nanmean(dx ** 2), np.nanmean(dy ** 2)
        mx = (ax2 - ax_ ** 2) * pixel_size ** 2      # drift-subtracted
        my = (ay2 - ay_ ** 2) * pixel_size ** 2

        t_lag.append(i * frame_interval)
        mean_dx.append(ax_ * pixel_size)
        msd_x.append(mx)
        msd_y.append(my)
        msd_2d.append(mx + my)
        n_list.append(n)

    return t_lag, mean_dx, msd_x, msd_y, msd_2d, n_list


def write_msd_csv(
    tracks: Dict[int, List[Tuple[int, float, float]]],
    n_frames: int,
    pixel_size: float,
    frame_interval: float,
    out_path: str,
    nmin: int = 10,
) -> None:
    """Compute the MSD table and write it to CSV."""
    import csv

    t_lag, mean_dx, msd_x, msd_y, msd_2d, n_list = compute_msd(
        tracks, n_frames, pixel_size, frame_interval, nmin
    )
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_lag_s", "mean_dx_um", "MSDx_um2",
                    "MSDy_um2", "MSD_2D_um2", "n"])
        for row in zip(t_lag, mean_dx, msd_x, msd_y, msd_2d, n_list):
            w.writerow(row)

# 5. Visualize trajectories

def plot_tracks(
    tracks: Dict[int, List[Tuple[int, float, float]]],
    stack_shape: Tuple[int, int, int],
    out_path: str,
    min_len_to_draw: int = 2,
) -> None:
    """
    Draw each trajectory as a line over the field of view and save to PNG.
    Longer tracks are drawn on top and brighter, so real particles stand out
    from short noise fragments.
    """
    import matplotlib.pyplot as plt

    _, h, w = stack_shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # image coords: y increases downward
    ax.set_aspect("equal")
    ax.set_title(f"{len(tracks)} trajectories")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    # sort so long tracks draw last (on top)
    ordered = sorted(tracks.items(), key=lambda kv: len(kv[1]))
    for _tid, pts in ordered:
        if len(pts) < min_len_to_draw:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        # longer tracks more opaque
        alpha = min(1.0, 0.2 + len(pts) / 20.0)
        ax.plot(xs, ys, linewidth=0.8, alpha=alpha)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)