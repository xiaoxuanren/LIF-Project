import os

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - environment dependent import.
    h5py = None


def _record_has_key(record, key):
    """Check whether a record-like object contains a requested field."""
    if hasattr(record, "files"):
        return key in record.files
    return key in record


def _coerce_scalar(value):
    """Collapse NumPy scalar containers into regular Python values."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _coerce_str(value):
    """Normalize stored string metadata from npz/hdf5 containers."""
    value = _coerce_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_column_selection(selection):
    """Convert boolean column masks into h5py-friendly slices or index lists."""
    if isinstance(selection, np.ndarray) and selection.dtype == bool:
        indices = np.flatnonzero(selection)
        if indices.size == 0:
            return slice(0, 0, 1)
        if np.all(np.diff(indices) == 1):
            return slice(int(indices[0]), int(indices[-1]) + 1, 1)
        return indices.tolist()
    return selection


class Hdf5VoltageTraceView:
    """Lazy read-only view over a chunked external HDF5 voltage dataset."""

    def __init__(self, file_path, dataset_name="voltage_traces"):
        if h5py is None:
            raise ImportError(
                "h5py is required to load external HDF5 voltage traces. "
                "Install it with 'pip install h5py'."
            )

        self.file_path = file_path
        self.dataset_name = dataset_name
        self._handle = h5py.File(file_path, "r")
        self._dataset = self._handle[dataset_name]

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def dtype(self):
        return self._dataset.dtype

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            return self._dataset[row_key, _normalize_column_selection(col_key)]
        return self._dataset[key]

    def __array__(self, dtype=None):
        array = self._dataset[()]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def astype(self, dtype):
        return np.asarray(self, dtype=dtype)

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
            self._dataset = None

    def __del__(self):  # pragma: no cover - best-effort cleanup.
        self.close()


class ChunkedHdf5VoltageRecorder:
    """Stream full-dt voltage traces into a chunked external HDF5 dataset."""

    def __init__(
        self,
        file_path,
        n_neurons,
        n_samples,
        sample_rate_ms,
        simulation_dt_ms,
        chunk_samples=4096,
        dataset_name="voltage_traces",
    ):
        if h5py is None:
            raise ImportError(
                "h5py is required for chunked HDF5 voltage storage. "
                "Install it with 'pip install h5py'."
            )

        self.file_path = file_path
        self.dataset_name = dataset_name
        self.n_neurons = int(n_neurons)
        self.n_samples = int(n_samples)
        self.sample_rate_ms = float(sample_rate_ms)
        self.simulation_dt_ms = float(simulation_dt_ms)
        self.chunk_samples = max(1, min(int(chunk_samples), self.n_samples))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._handle = h5py.File(file_path, "w")
        self._dataset = self._handle.create_dataset(
            dataset_name,
            shape=(self.n_neurons, self.n_samples),
            dtype=np.float32,
            chunks=(self.n_neurons, self.chunk_samples),
            compression="lzf",
            shuffle=True,
        )
        self._dataset.attrs["units"] = "mV"
        self._dataset.attrs["sample_rate_ms"] = self.sample_rate_ms
        self._dataset.attrs["simulation_dt_ms"] = self.simulation_dt_ms
        self._dataset.attrs["storage_backend"] = "hdf5_external"

        self._buffer = np.empty((self.n_neurons, self.chunk_samples), dtype=np.float32)
        self._buffer_count = 0
        self._write_offset = 0

    def write_step(self, voltage_values):
        """Append one timestep of voltage values to the chunk buffer."""
        values = np.asarray(voltage_values, dtype=np.float32)
        if values.shape != (self.n_neurons,):
            raise ValueError(
                f"Expected voltage_values with shape ({self.n_neurons},), got {values.shape}"
            )

        self._buffer[:, self._buffer_count] = values
        self._buffer_count += 1
        if self._buffer_count == self._buffer.shape[1]:
            self.flush()

    def flush(self):
        """Flush any buffered samples into the HDF5 dataset."""
        if self._buffer_count == 0:
            return

        end_offset = self._write_offset + self._buffer_count
        self._dataset[:, self._write_offset:end_offset] = self._buffer[:, : self._buffer_count]
        self._write_offset = end_offset
        self._buffer_count = 0

    def finalize(self):
        """Flush buffered data, close the file, and return storage metadata."""
        self.flush()
        self._handle.flush()
        self.close()
        return {
            "storage_backend": "hdf5_external",
            "voltage_file": self.file_path,
            "voltage_dataset": self.dataset_name,
            "n_samples": self.n_samples,
            "dtype": "float32",
            "units": "mV",
        }

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
            self._dataset = None


def resolve_voltage_path(recording_path, stored_path):
    """Resolve a stored relative voltage sidecar path against one recording file."""
    stored_path = _coerce_str(stored_path)
    if os.path.isabs(stored_path) or recording_path is None:
        return stored_path
    return os.path.join(os.path.dirname(recording_path), stored_path)


def resolve_recording_voltage(record, recording_path=None, load_into_memory=False):
    """Resolve inline or external voltage storage into one common bundle."""
    if _record_has_key(record, "voltage_traces_raw"):
        traces = record["voltage_traces_raw"]
        source_key = "voltage_traces_raw"
        storage_backend = "inline_npz"
    elif _record_has_key(record, "voltage_traces"):
        traces = record["voltage_traces"]
        source_key = "voltage_traces"
        storage_backend = _coerce_str(record["voltage_storage_backend"]) if _record_has_key(record, "voltage_storage_backend") else "inline_npz"
    elif _record_has_key(record, "voltage_storage_backend"):
        storage_backend = _coerce_str(record["voltage_storage_backend"])
        if storage_backend != "hdf5_external":
            raise ValueError(f"Unsupported voltage storage backend: {storage_backend}")

        voltage_file = resolve_voltage_path(recording_path, record["voltage_hdf5_file"])
        dataset_name = _coerce_str(record["voltage_hdf5_dataset"]) if _record_has_key(record, "voltage_hdf5_dataset") else "voltage_traces"
        traces = Hdf5VoltageTraceView(voltage_file, dataset_name=dataset_name)
        source_key = "voltage_hdf5_file"
    else:
        return None

    if load_into_memory:
        traces = np.asarray(traces, dtype=np.float32)

    if _record_has_key(record, "voltage_times"):
        times = np.asarray(record["voltage_times"], dtype=np.float32)
    elif _record_has_key(record, "voltage_sample_rate") and _record_has_key(record, "voltage_n_samples"):
        sample_rate = float(_coerce_scalar(record["voltage_sample_rate"]))
        n_samples = int(_coerce_scalar(record["voltage_n_samples"]))
        times = np.arange(n_samples, dtype=np.float32) * np.float32(sample_rate)
    else:
        times = None

    voltage_bundle = {
        "traces": traces,
        "times": times,
        "sample_rate": float(_coerce_scalar(record["voltage_sample_rate"])) if _record_has_key(record, "voltage_sample_rate") else None,
        "storage_backend": storage_backend,
        "source_key": source_key,
        "units": _coerce_str(record["voltage_units"]) if _record_has_key(record, "voltage_units") else "mV",
    }
    if _record_has_key(record, "voltage_n_samples"):
        voltage_bundle["n_samples"] = int(_coerce_scalar(record["voltage_n_samples"]))
    elif times is not None:
        voltage_bundle["n_samples"] = int(len(times))
    return voltage_bundle