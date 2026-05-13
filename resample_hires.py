"""
Resample all saved recordings: replace resampled_spikes with 0.1 ms resolution binary spike train.

Overwrites each recording .npz in-place, keeping all existing keys but replacing:
  - resampled_spikes -> binary matrix at dt=0.1 ms
  - resampling_frequency -> 10000 (Hz)
  - resampling_interval_ms -> 0.1
  - resampled_time_points -> time axis at 0.1 ms steps
  - resampled_spike_positions -> indices where spikes occurred
"""

import numpy as np
import os
import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIF data")
DT = 0.1  # ms
TARGET_FREQ = 1000.0 / DT  # 10000 Hz


def resample_hires(spike_times, duration, dt=DT):
    """Create binary spike train at dt resolution."""
    n_neurons = len(spike_times)
    n_bins = int(duration / dt)

    resampled_spikes = np.zeros((n_neurons, n_bins), dtype=np.int8)
    for i in range(n_neurons):
        for t in spike_times[i]:
            bin_idx = int(t / dt)
            if 0 <= bin_idx < n_bins:
                resampled_spikes[i, bin_idx] = 1

    resampled_time_points = np.arange(n_bins) * dt
    resampled_spike_positions = np.argwhere(resampled_spikes)

    return resampled_spikes, resampled_time_points, resampled_spike_positions


def main():
    sessions = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
    if not sessions:
        print("No sessions found in", DATA_DIR)
        return

    for session_dir in sessions:
        if not os.path.isdir(session_dir):
            continue
        session_name = os.path.basename(session_dir)
        rec_files = sorted(glob.glob(os.path.join(session_dir, "recording[0-9][0-9][0-9].npz")))

        if not rec_files:
            continue

        print(f"\nSession: {session_name}  ({len(rec_files)} recordings)")

        for rec_path in rec_files:
            rec_name = os.path.basename(rec_path)
            print(f"  {rec_name} ...", end=" ", flush=True)

            # Load all existing data
            data = np.load(rec_path, allow_pickle=True)
            save_dict = {key: data[key] for key in data.files}

            spike_times = data['spike_times']
            duration = float(data['duration'])

            # Resample at 0.1 ms
            resampled_spikes, resampled_time_points, resampled_spike_positions = \
                resample_hires(spike_times, duration)

            # Replace resampled fields
            save_dict['resampled_spikes'] = resampled_spikes
            save_dict['resampled_time_points'] = resampled_time_points
            save_dict['resampled_spike_positions'] = resampled_spike_positions
            save_dict['resampling_frequency'] = TARGET_FREQ
            save_dict['resampling_interval_ms'] = DT

            # Save back (overwrite)
            np.savez_compressed(rec_path, **save_dict)

            n_spikes = int(resampled_spikes.sum())
            file_size = os.path.getsize(rec_path) / (1024 * 1024)
            print(f"done  ({resampled_spikes.shape[0]} neurons x {resampled_spikes.shape[1]} bins, "
                  f"{n_spikes} spikes, {file_size:.1f} MB)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
