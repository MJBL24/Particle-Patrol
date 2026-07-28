import os, threading

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use(Path(__file__).resolve().parents[3] / 'utils' / 'presentation.mplstyle')

from gui.config import PreviewConfigGUI, InputConfigGUI, BarcodeConfigGUI
from utils.gui import create_popup, create_option_section

from analysis.tracking import build_binary_stack, temporal_overlap_fraction, link_by_connectivity

# how many frames of the sample to use for the live preview (a slice, for speed)
PREVIEW_FRAMES = 25

def create_tracking_frame(
    parent,
    config: BarcodeConfigGUI,
    preview_config: PreviewConfigGUI,
    input_config: InputConfigGUI,
):
    """Create the particle-tracking settings tab with a live trajectory preview."""
    frame = ttk.Frame(parent)

    ct = config.tracking_parameters      # tracking settings (GUI vars)
    cp = preview_config
    ci = input_config

    row = 0


    # weighted centroid toggle
    create_option_section(
        frame, row, ct.weighted_centroid,
        "Weighted Centroid (sub-pixel)",
        "Use the intensity-weighted centroid (sits on the particle's bright peak) "
        "instead of the binary centroid. More accurate, requires Binning = 1."
    )
    row += 1

    # threshold offset
    thr_label = tk.Label(frame, text="Binarization Threshold:")
    thr_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Cutoff for binarization. Lower = fatter blobs (more overlap, more noise); "
                        "higher = tighter blobs. Watch the preview.", row, thr_label)
    thr_spin = ttk.Spinbox(frame, from_=-1.0, to=1.0, increment=0.025,
                           textvariable=ct.threshold_offset, format="%.3f", width=7)
    thr_spin.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # ---- binning ----
    bin_label = tk.Label(frame, text="Binning Ratio:")
    bin_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Downsample factor before binarization. Use 1 for weighted centroid "
                        "and for correct pixel-size scaling.", row, bin_label)
    bin_menu = ttk.Combobox(frame, textvariable=ct.bin_factor,
                            values=[1, 2, 4, 8], width=5, state="readonly")
    bin_menu.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # minimum particle size (per-frame pixels)
    mps_label = tk.Label(frame, text="Minimum Particle Size (px):")
    mps_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Drop islands smaller than this many pixels in a single frame "
                        "(removes background-noise specks).", row, mps_label)
    mps_spin = ttk.Spinbox(frame, from_=0, to=500, increment=1,
                           textvariable=ct.min_particle_size, width=7)
    mps_spin.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # minimum track length (frames)
    mtl_label = tk.Label(frame, text="Minimum Track Length (frames):")
    mtl_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Drop trajectories that persist for fewer than this many frames "
                        "(removes flickering noise).", row, mtl_label)
    mtl_spin = ttk.Spinbox(frame, from_=1, to=100, increment=1,
                           textvariable=ct.min_track_length, width=7)
    mtl_spin.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # time gap
    tg_label = tk.Label(frame, text="Time Gap (frames):")
    tg_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Bridge a particle that disappears for up to this many frames "
                        "before calling its return a new particle.", row, tg_label)
    tg_spin = ttk.Spinbox(frame, from_=0, to=20, increment=1,
                          textvariable=ct.time_gap, width=7)
    tg_spin.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # spatial dilation
    sd_label = tk.Label(frame, text="Spatial Dilation (px):")
    sd_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Grow blobs by this many pixels so they can still link across a "
                        "larger jump. Helps sparse fast data; merges dense particles.", row, sd_label)
    sd_spin = ttk.Spinbox(frame, from_=0, to=20, increment=1,
                          textvariable=ct.spatial_dilation, width=7)
    sd_spin.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # overlap threshold (the good/poor cutoff)
    ot_label = tk.Label(frame, text="Overlap Threshold (good/poor):")
    ot_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
    create_popup(frame, "Minimum temporal overlap fraction to call the video a GOOD candidate "
                        "for connectivity tracking.", row, ot_label)
    ot_spin = ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05,
                          textvariable=ct.overlap_threshold, format="%.2f", width=7)
    ot_spin.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # the good/poor candidate message
    candidate_label = tk.Label(frame, text="Upload a file to check tracking feasibility.",
                               font=("Segoe UI", 10, "bold"))
    candidate_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 5))
    row += 1

    # trajectory preview figure
    preview_title = tk.Label(frame, text="Trajectory Preview (first %d frames)" % PREVIEW_FRAMES)
    preview_title.grid(row=row, column=0, columnspan=2, padx=5, pady=(10, 2), sticky="w")
    row += 1

    fig_traj = Figure(figsize=(4, 4), facecolor="white")
    ax_traj = fig_traj.add_subplot(111)
    ax_traj.set_facecolor("white")
    ax_traj.axis("off")
    canvas_traj = FigureCanvasTkAgg(fig_traj, master=frame)
    canvas_traj.draw()
    canvas_traj.get_tk_widget().grid(row=row, column=0, columnspan=2, padx=5, pady=(10, 5))
    fig_traj.tight_layout()
    row += 1

    # core preview computation
    def compute_and_draw(*args):
        # only compute once a real file is loaded
        try:
            frame_data = cp.sample_preview   # (frames, y, x, channels)
        except Exception:
            frame_data = None

        file_loaded = bool(ci.file_path.get()) or bool(cp.sample_file.get())
        if (not file_loaded) or frame_data is None or len(frame_data) == 0:
            candidate_label.config(text="Upload a file to check tracking feasibility.",
                                   fg="black")
            ax_traj.clear(); ax_traj.set_facecolor("white"); ax_traj.axis("off")
            canvas_traj.draw()
            return

        # channel
        try:
            if config.channels.parse_all_channels.get():
                channel = 0
            else:
                channel = config.channels.selected_channel.get()
        except Exception:
            channel = 0

        n = min(PREVIEW_FRAMES, len(frame_data))
        video_slice = frame_data[:n, :, :, channel]

        # read settings (fall back if a field is mid-edit)
        try:
            offset = ct.threshold_offset.get()
            bin_factor = ct.bin_factor.get()
            invert = ct.invert_binarization.get()
            min_particle = ct.min_particle_size.get()
            min_track = ct.min_track_length.get()
            min_comp = ct.min_component_size.get()
            conn = ct.connectivity.get()
            tgap = ct.time_gap.get()
            sdil = ct.spatial_dilation.get()
            thr_overlap = ct.overlap_threshold.get()
            weighted = ct.weighted_centroid.get()
        except tk.TclError:
            return

        try:
            stack = build_binary_stack(video_slice, offset, bin_factor,
                                       invert=invert, min_particle_size=min_particle)
            overlap = temporal_overlap_fraction(stack)

            if overlap >= thr_overlap:
                candidate_label.config(
                    text=f"\u2713 GOOD candidate for connectivity tracking "
                         f"(overlap {overlap:.3f})", fg="green")
            else:
                candidate_label.config(
                    text=f"\u2717 POOR candidate \u2014 particles move too far "
                         f"(overlap {overlap:.3f}). Consider the detect-then-link method.",
                    fg="red")

            intensity = video_slice if (weighted and bin_factor == 1 and
                                        video_slice.shape == stack.shape) else None
            tracks = link_by_connectivity(
                stack, connectivity=conn, time_gap=tgap,
                min_track_length=min_track, min_component_size=min_comp,
                spatial_dilation=sdil, intensity_stack=intensity,
                weighted=(intensity is not None),
            )

            # draw -- bigger, clearer trajectories
            ax_traj.clear()
            ax_traj.set_facecolor("white")
            _, h, w = stack.shape
            ax_traj.set_xlim(0, w); ax_traj.set_ylim(h, 0)
            ax_traj.set_aspect("equal"); ax_traj.axis("off")
            ax_traj.set_title(f"{len(tracks)} trajectories", fontsize=9)
            for pts in tracks.values():
                if len(pts) < 1:
                    continue
                xs = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                # thicker line + visible markers so slow/short tracks are easy to see
                ax_traj.plot(xs, ys, linewidth=1.8, marker='o',
                             markersize=4, alpha=0.9)
            fig_traj.tight_layout()
            canvas_traj.draw()
        except Exception as e:
            candidate_label.config(text=f"Preview error: {e}", fg="red")

    # debounce so dragging spinboxes doesn't recompute on every keystroke
    pending = {"id": None}

    def schedule(*args):
        if pending["id"] is not None:
            frame.after_cancel(pending["id"])
        pending["id"] = frame.after(400, compute_and_draw)

    for var in (ct.threshold_offset, ct.bin_factor, ct.invert_binarization,
                ct.min_particle_size, ct.min_track_length, ct.min_component_size,
                ct.connectivity, ct.time_gap, ct.spatial_dilation,
                ct.overlap_threshold, ct.weighted_centroid):
        var.trace_add("write", schedule)

    ci.file_path.trace_add("write", schedule)
    cp.sample_file.trace_add("write", schedule)
    config.channels.selected_channel.trace_add("write", schedule)
    config.channels.parse_all_channels.trace_add("write", schedule)

    def sync_sample_file(*args):
        p = ci.file_path.get()
        if p and cp.sample_file.get() != p:
            cp.sample_file.set(p)

    ci.file_path.trace_add("write", sync_sample_file)
    sync_sample_file()

    compute_and_draw()

    return frame