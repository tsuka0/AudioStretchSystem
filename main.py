import os
import subprocess
import tkinter as tk
from tkinter import filedialog
import threading
import queue
import numpy as np
import soundfile as sf
import pyworld as pw
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import sounddevice as sd

STRETCH = 10.0
FRAME_PERIOD = 2.5
ANALYSIS_MARGIN = 0.15

def select_input(root):
    return filedialog.askopenfilename(
        parent=root,
        title="Select Input Audio File",
        filetypes=[
            ("Audio Files", "*.wav *.flac *.aiff *.aif"),
            ("WAV", "*.wav"),
            ("All Files", "*.*")
        ]
    )

def select_output(root, initial_name):
    return filedialog.asksaveasfilename(
        parent=root,
        title="Select Output File Location",
        initialfile=initial_name,
        defaultextension=".wav",
        filetypes=[
            ("WAV", "*.wav"),
            ("All Files", "*.*")
        ]
    )

def stretch_selected(x, fs, start, end):
    margin = int(fs * ANALYSIS_MARGIN)

    analysis_start = max(
        0,
        start - margin
    )

    analysis_end = min(
        len(x),
        end + margin
    )

    analysis = x[
        analysis_start:analysis_end
    ]

    if len(analysis) < int(fs * 0.03):
        return None

    f0, t = pw.dio(
        analysis,
        fs,
        frame_period=FRAME_PERIOD
    )

    f0 = pw.stonemask(
        analysis,
        f0,
        t,
        fs
    )

    sp = pw.cheaptrick(
        analysis,
        f0,
        t,
        fs
    )

    ap = pw.d4c(
        analysis,
        f0,
        t,
        fs
    )

    local_start = (
        start - analysis_start
    ) / fs

    local_end = (
        end - analysis_start
    ) / fs

    mask = (
        (t >= local_start) &
        (t <= local_end)
    )

    indexes = np.where(mask)[0]

    if len(indexes) < 2:
        return None

    source_t = t[indexes]

    f0_part = f0[indexes]
    sp_part = sp[indexes]
    ap_part = ap[indexes]

    duration = (
        end - start
    ) / fs

    output_frames = max(
        2,
        int(
            duration *
            STRETCH /
            (FRAME_PERIOD / 1000.0)
        )
    )

    target_t = np.linspace(
        source_t[0],
        source_t[-1],
        output_frames
    )

    f0_new = np.interp(
        target_t,
        source_t,
        f0_part
    )

    sp_new = np.empty(
        (
            output_frames,
            sp_part.shape[1]
        ),
        dtype=np.float64
    )

    ap_new = np.empty(
        (
            output_frames,
            ap_part.shape[1]
        ),
        dtype=np.float64
    )

    for i in range(
        sp_part.shape[1]
    ):
        sp_new[:, i] = np.interp(
            target_t,
            source_t,
            sp_part[:, i]
        )

    for i in range(
        ap_part.shape[1]
    ):
        ap_new[:, i] = np.interp(
            target_t,
            source_t,
            ap_part[:, i]
        )

    y = pw.synthesize(
        f0_new,
        sp_new,
        ap_new,
        fs
    )

    target_samples = int(
        duration *
        STRETCH *
        fs
    )

    if len(y) > target_samples:
        y = y[:target_samples]

    elif len(y) < target_samples:
        y = np.pad(
            y,
            (
                0,
                target_samples - len(y)
            )
        )

    peak = np.max(
        np.abs(y)
    )

    if peak > 0.98:
        y = (
            y / peak
        ) * 0.98

    return y

def select_region(x, fs):
    duration = len(x) / fs
    time = np.arange(len(x)) / fs

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.12
    )

    ax.plot(
        time,
        x,
        linewidth=0.6
    )

    ax.set_title(
        "Drag: Select Range / Space: Analyze & Preview / Enter: Confirm / Esc: Cancel"
    )

    ax.set_xlabel(
        "Time (sec)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_xlim(
        0,
        duration
    )

    selected_range = [None, None]
    patch = [None]
    processing = [False]
    playing = [False]
    result_queue = queue.Queue()

    status_text = fig.text(
        0.5,
        0.02,
        "Select a range",
        ha="center",
        va="bottom",
        fontsize=12
    )

    def set_status(text):
        status_text.set_text(text)
        fig.canvas.draw_idle()

    def on_select(xmin, xmax):
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        selected_range[0] = max(
            0.0,
            xmin
        )

        selected_range[1] = min(
            duration,
            xmax
        )

        if patch[0] is not None:
            patch[0].remove()

        patch[0] = ax.axvspan(
            selected_range[0],
            selected_range[1],
            alpha=0.3
        )

        set_status(
            "Range selected - Press Space to analyze and preview"
        )

    def worker(start, end):
        try:
            result = stretch_selected(
                x,
                fs,
                start,
                end
            )

            result_queue.put(
                (
                    "success",
                    result
                )
            )

        except Exception as e:
            result_queue.put(
                (
                    "error",
                    e
                )
            )

    def check_result():
        try:
            result_type, result = result_queue.get_nowait()

            processing[0] = False

            if result_type == "success":
                if result is None:
                    set_status(
                        "Analysis failed"
                    )
                    return

                set_status(
                    "Analysis complete! Press Space to play"
                )

                preview_data = result

                try:
                    sd.play(
                        preview_data,
                        fs
                    )

                    playing[0] = True

                    def wait_playback():
                        sd.wait()

                        def finished():
                            playing[0] = False

                            if plt.fignum_exists(
                                fig.number
                            ):
                                set_status(
                                    "Analysis complete! Press Space to play"
                                )

                        fig.canvas.get_tk_widget().after(
                            0,
                            finished
                        )

                    threading.Thread(
                        target=wait_playback,
                        daemon=True
                    ).start()

                except Exception:
                    playing[0] = False
                    set_status(
                        "Playback failed"
                    )

            else:
                set_status(
                    "Analysis failed"
                )

                print(
                    "Preview error:",
                    result
                )

        except queue.Empty:
            pass

        if plt.fignum_exists(
            fig.number
        ):
            fig.canvas.get_tk_widget().after(
                30,
                check_result
            )

    def preview():
        if selected_range[0] is None:
            set_status(
                "Please select a range first"
            )
            return

        if processing[0]:
            return

        if playing[0]:
            sd.stop()
            playing[0] = False

            set_status(
                "Analysis complete! Press Space to play"
            )

            return

        start = max(
            0,
            int(
                selected_range[0] *
                fs
            )
        )

        end = min(
            len(x),
            int(
                selected_range[1] *
                fs
            )
        )

        if end <= start:
            return

        processing[0] = True

        set_status(
            "Analyzing..."
        )

        thread = threading.Thread(
            target=worker,
            args=(
                start,
                end
            ),
            daemon=True
        )

        thread.start()

    def on_key(event):
        if event.key == " ":
            preview()

        elif event.key == "enter":
            sd.stop()
            playing[0] = False
            plt.close(fig)

        elif event.key == "escape":
            sd.stop()
            playing[0] = False

            selected_range[0] = None
            selected_range[1] = None

            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )

    selector = SpanSelector(
        ax,
        on_select,
        "horizontal",
        useblit=True,
        props=dict(
            alpha=0.3
        )
    )

    fig.canvas.get_tk_widget().after(
        30,
        check_result
    )

    plt.show()

    if (
        selected_range[0] is None or
        selected_range[1] is None
    ):
        return None

    return (
        selected_range[0],
        selected_range[1]
    )

def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes(
        "-topmost",
        True
    )

    try:
        input_file = select_input(
            root
        )

        if not input_file:
            return

        x, fs = sf.read(
            input_file,
            always_2d=False
        )

        if x.ndim > 1:
            x = np.mean(
                x,
                axis=1
            )

        x = x.astype(
            np.float64
        )

        region = select_region(
            x,
            fs
        )

        if region is None:
            return

        start_time, end_time = region

        start = max(
            0,
            int(
                start_time *
                fs
            )
        )

        end = min(
            len(x),
            int(
                end_time *
                fs
            )
        )

        if end <= start:
            return

        output_name = (
            os.path.splitext(
                os.path.basename(
                    input_file
                )
            )[0]
            + "_long.wav"
        )

        output_file = select_output(
            root,
            output_name
        )

        if not output_file:
            return

        output = stretch_selected(
            x,
            fs,
            start,
            end
        )

        if output is None:
            return

        sf.write(
            output_file,
            output,
            fs,
            subtype="FLOAT"
        )

        output_folder = os.path.dirname(
            os.path.abspath(
                output_file
            )
        )

    finally:
        root.destroy()

if __name__ == "__main__":
    main()