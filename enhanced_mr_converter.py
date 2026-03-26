"""
Enhanced MR → Classic DICOM Converter
Converts Enhanced MR Image Storage (1.2.840.10008.5.1.4.1.1.4.1) multi-frame files
into Classic MR Image Storage (1.2.840.10008.5.1.4.1.1.4) single-frame files.

Usage: python enhanced_mr_converter.py
Dependencies: pydicom, numpy (pip), tkinter (stdlib)
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import ttk

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENHANCED_MR_SOP_CLASS = '1.2.840.10008.5.1.4.1.1.4.1'
CLASSIC_MR_SOP_CLASS  = '1.2.840.10008.5.1.4.1.1.4'
REALIZE_UID_PREFIX    = '1.2.826.0.1.3680043.10.501.'
EXPLICIT_LITTLE_ENDIAN = ExplicitVRLittleEndian  # '1.2.840.10008.1.2.1'

# Top-level tags to copy straight from the source dataset (if present).
# Grouped by DICOM standard section for readability.
COPY_TAGS = [
    # --- Patient ---
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1030),  # PatientWeight
    # --- General Study ---
    (0x0008, 0x0020),  # StudyDate
    (0x0008, 0x0021),  # SeriesDate
    (0x0008, 0x0022),  # AcquisitionDate
    (0x0008, 0x0030),  # StudyTime
    (0x0008, 0x0031),  # SeriesTime
    (0x0008, 0x0032),  # AcquisitionTime
    (0x0008, 0x0050),  # AccessionNumber
    (0x0008, 0x0060),  # Modality
    (0x0008, 0x0070),  # Manufacturer
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x1030),  # StudyDescription
    (0x0008, 0x103e),  # SeriesDescription
    (0x0008, 0x1090),  # ManufacturerModelName
    (0x0020, 0x000d),  # StudyInstanceUID
    (0x0020, 0x0010),  # StudyID
    (0x0020, 0x0011),  # SeriesNumber
    (0x0020, 0x0052),  # FrameOfReferenceUID
    # --- MR Image ---
    (0x0018, 0x0081),  # EchoTime
    (0x0018, 0x0082),  # InversionTime
    (0x0018, 0x0083),  # NumberOfAverages
    (0x0018, 0x0084),  # ImagingFrequency
    (0x0018, 0x0085),  # ImagedNucleus
    (0x0018, 0x0087),  # MagneticFieldStrength
    (0x0018, 0x0091),  # EchoTrainLength
    (0x0018, 0x0093),  # PercentSampling
    (0x0018, 0x0094),  # PercentPhaseFieldOfView
    (0x0018, 0x0095),  # PixelBandwidth
    (0x0018, 0x1000),  # DeviceSerialNumber
    (0x0018, 0x1020),  # SoftwareVersions
    (0x0018, 0x1030),  # ProtocolName
    (0x0018, 0x1088),  # HeartRate
    (0x0018, 0x1094),  # TriggerWindow
    (0x0018, 0x1250),  # ReceiveCoilName
    (0x0018, 0x1310),  # AcquisitionMatrix
    (0x0018, 0x1312),  # InPlanePhaseEncodingDirection
    (0x0018, 0x1314),  # FlipAngle
    (0x0018, 0x1316),  # SAR
    (0x0018, 0x5100),  # PatientPosition
    (0x0018, 0x9089),  # DiffusionGradientOrientation (if present, harmless)
    # --- Image Plane ---
    (0x0028, 0x0010),  # Rows
    (0x0028, 0x0011),  # Columns
    (0x0028, 0x0030),  # PixelSpacing  (may be overridden from SharedFG)
    (0x0028, 0x0100),  # BitsAllocated
    (0x0028, 0x0101),  # BitsStored
    (0x0028, 0x0102),  # HighBit
    (0x0028, 0x0103),  # PixelRepresentation
    (0x0028, 0x1050),  # WindowCenter
    (0x0028, 0x1051),  # WindowWidth
    (0x0028, 0x1052),  # RescaleIntercept
    (0x0028, 0x1053),  # RescaleSlope
    (0x0028, 0x1054),  # RescaleType
    # --- MR-specific acquisition params ---
    (0x0018, 0x0023),  # MRAcquisitionType
    (0x0018, 0x0080),  # RepetitionTime
    (0x0018, 0x0050),  # SliceThickness
    (0x0018, 0x0089),  # NumberOfPhaseEncodingSteps
]

# Tags to explicitly remove from the output (Enhanced MR artefacts)
REMOVE_TAGS = [
    (0x0028, 0x0008),  # NumberOfFrames
    (0x5200, 0x9229),  # SharedFunctionalGroupsSequence
    (0x5200, 0x9230),  # PerFrameFunctionalGroupsSequence
    (0x0020, 0x9221),  # DimensionOrganizationSequence
    (0x0020, 0x9222),  # DimensionIndexSequence
    (0x0020, 0x9157),  # DimensionIndexValues
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_slice_location(ipp: list, iop: list) -> float:
    """
    Compute the scalar SliceLocation from ImagePositionPatient and
    ImageOrientationPatient using the cross-product of row/col vectors.
    slice_normal = row_vec × col_vec
    SliceLocation = slice_normal · ipp
    """
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    normal = np.cross(row, col)
    return float(np.dot(normal, np.array(ipp, dtype=float)))


def extract_iop(ds: pydicom.Dataset) -> list:
    """
    Extract ImageOrientationPatient from SharedFunctionalGroupsSequence
    > PlaneOrientationSequence, falling back to the top-level tag.
    Returns a list of 6 floats, or None if not found.
    """
    try:
        shared = ds.SharedFunctionalGroupsSequence[0]
        plane_orient = shared.PlaneOrientationSequence[0]
        return [float(v) for v in plane_orient.ImageOrientationPatient]
    except (AttributeError, IndexError, KeyError):
        pass
    if hasattr(ds, 'ImageOrientationPatient'):
        return [float(v) for v in ds.ImageOrientationPatient]
    try:
        frame0 = ds.PerFrameFunctionalGroupsSequence[0]
        po = frame0.PlaneOrientationSequence[0]
        return [float(v) for v in po.ImageOrientationPatient]
    except (AttributeError, IndexError, KeyError):
        pass
    return None


def extract_pixel_spacing(ds: pydicom.Dataset):
    """
    Extract PixelSpacing from SharedFunctionalGroupsSequence
    > PixelMeasuresSequence, falling back to top-level tag.
    Returns pydicom value or None.
    """
    try:
        shared = ds.SharedFunctionalGroupsSequence[0]
        pms = shared.PixelMeasuresSequence[0]
        return pms.PixelSpacing
    except (AttributeError, IndexError, KeyError):
        pass
    if hasattr(ds, 'PixelSpacing'):
        return ds.PixelSpacing
    try:
        frame0 = ds.PerFrameFunctionalGroupsSequence[0]
        pm = frame0[0x0028, 0x9110][0]
        return pm.PixelSpacing
    except (AttributeError, IndexError, KeyError):
        pass
    return None


def extract_slice_thickness(ds: pydicom.Dataset):
    """
    Extract SliceThickness from SharedFG, per-frame, or top-level.
    Returns float or None.
    """
    try:
        return float(ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
    except Exception:
        pass
    try:
        return float(ds.PerFrameFunctionalGroupsSequence[0][0x0028, 0x9110][0].SliceThickness)
    except Exception:
        pass
    if hasattr(ds, 'SliceThickness'):
        return float(ds.SliceThickness)
    return None


def scan_source_folder(folder: str) -> dict:
    """
    Peek at the first .dcm file in folder to verify it is Enhanced MR,
    then count files and frames.
    Returns dict with keys: valid (bool), n_files, n_frames, total, error.
    """
    dcm_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith('.dcm')
    ]
    if not dcm_files:
        return {'valid': False, 'error': 'No .dcm files found in folder.'}

    first = os.path.join(folder, dcm_files[0])
    try:
        ds = pydicom.dcmread(first, stop_before_pixels=True)
    except Exception as e:
        return {'valid': False, 'error': f'Cannot read DICOM: {e}'}

    sop = getattr(ds, 'SOPClassUID', None)
    if sop is None or str(sop) != ENHANCED_MR_SOP_CLASS:
        sop_name = getattr(sop, 'name', str(sop)) if sop else 'unknown'
        return {
            'valid': False,
            'error': f'Not Enhanced MR. Found SOP class: {sop_name}'
        }

    n_frames = int(getattr(ds, 'NumberOfFrames', 1))
    n_files  = len(dcm_files)
    return {
        'valid'  : True,
        'n_files': n_files,
        'n_frames': n_frames,
        'total'  : n_files * n_frames,
        'error'  : None,
    }


def load_enhanced_mr_files(folder: str, log_cb) -> tuple:
    """
    Read all Enhanced MR .dcm files in folder.
    Returns (list_of_frame_dicts, first_source_dataset).

    Each frame_dict contains:
      in_stack_pos     : int   (1-based slice index)
      temporal_pos     : int   (1-based phase index)
      cardiac_delay_ms : float (ms delay from R-wave)
      image_position   : list[float] (3 values, ImagePositionPatient)
      pixel_2d         : np.ndarray  (2-D uint16 array)
    """
    dcm_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith('.dcm')
    )
    frames = []
    src_ds = None  # first dataset, kept as template for top-level tags

    for fname in dcm_files:
        fpath = os.path.join(folder, fname)
        log_cb(f'  Reading {fname}')
        ds = pydicom.dcmread(fpath)

        if src_ds is None:
            src_ds = ds

        # Decode pixel data once per file — avoids 24× redundant decompression
        try:
            pixel_volume = ds.pixel_array  # shape: (n_frames, rows, cols)
        except Exception as e:
            log_cb(f'    WARNING: Cannot decode pixel data in {fname}: {e}')
            pixel_volume = None

        per_frame_seq = ds.PerFrameFunctionalGroupsSequence

        for i, frame_fg in enumerate(per_frame_seq):
            # --- Temporal info ---
            temporal_pos     = None
            cardiac_delay_ms = 0.0
            try:
                fc = frame_fg.FrameContentSequence[0]
                temporal_pos = int(fc.TemporalPositionIndex)
            except (AttributeError, IndexError):
                pass
            try:
                ct = frame_fg[0x0018, 0x9118][0]
                cardiac_delay_ms = float(ct[0x0020, 0x9153].value)
            except (KeyError, IndexError):
                pass

            # --- Spatial info ---
            in_stack_pos  = None
            image_position = None
            try:
                fc = frame_fg.FrameContentSequence[0]
                in_stack_pos = int(fc.InStackPositionNumber)
            except (AttributeError, IndexError):
                pass
            try:
                pp = frame_fg.PlanePositionSequence[0]
                image_position = [float(v) for v in pp.ImagePositionPatient]
            except (AttributeError, IndexError):
                pass

            # Fallback: derive from file position if tags are missing
            if temporal_pos is None:
                temporal_pos = i + 1
            if in_stack_pos is None:
                # Try DimensionIndexValues [temporal, stack, in_stack]
                try:
                    div = frame_fg.FrameContentSequence[0].DimensionIndexValues
                    in_stack_pos = int(div[2])
                except Exception:
                    in_stack_pos = i + 1

            pixel_2d = pixel_volume[i] if pixel_volume is not None else None

            frames.append({
                'in_stack_pos'    : in_stack_pos,
                'temporal_pos'    : temporal_pos,
                'cardiac_delay_ms': cardiac_delay_ms,
                'image_position'  : image_position,
                'pixel_2d'        : pixel_2d,
            })

    return frames, src_ds


def _build_output_dataset(frame: dict, src_ds: pydicom.Dataset,
                          series_uid: str, n_slices: int,
                          n_phases: int, iop: list) -> FileDataset:
    """
    Build a single Classic MR FileDataset from one extracted frame.
    """
    in_stack_pos     = frame['in_stack_pos']
    temporal_pos     = frame['temporal_pos']
    cardiac_delay_ms = frame['cardiac_delay_ms']
    image_position   = frame['image_position']
    pixel_2d         = frame['pixel_2d']

    instance_number = (temporal_pos - 1) * n_slices + in_stack_pos

    # --- File meta ---
    sop_instance_uid = generate_uid(REALIZE_UID_PREFIX)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = CLASSIC_MR_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID          = EXPLICIT_LITTLE_ENDIAN

    # --- Build dataset from scratch (avoid carrying over Enhanced MR sequences) ---
    ds = FileDataset(filename_or_obj=None,
                     dataset={},
                     file_meta=file_meta,
                     is_implicit_VR=False,
                     is_little_endian=True)
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    # Copy safe top-level tags from source
    for tag in COPY_TAGS:
        if tag in src_ds:
            try:
                ds[tag] = src_ds[tag]
            except Exception:
                pass  # skip unreadable tags silently

    # --- Overwrite / set required tags ---
    ds.SOPClassUID    = CLASSIC_MR_SOP_CLASS
    ds.SOPInstanceUID = sop_instance_uid

    # Series: new shared UID across all output files
    ds.SeriesInstanceUID = series_uid

    # Instance ordering
    ds.InstanceNumber = instance_number

    # Spatial
    if image_position:
        ds.ImagePositionPatient = [str(v) for v in image_position]
        if iop:
            ds.SliceLocation = compute_slice_location(image_position, iop)

    # Orientation (from SharedFG or top-level of source)
    if iop:
        ds.ImageOrientationPatient = [str(v) for v in iop]

    # PixelSpacing from SharedFG (may already be in COPY_TAGS via top-level)
    ps = extract_pixel_spacing(src_ds)
    if ps is not None:
        ds.PixelSpacing = ps

    # SliceThickness
    st = extract_slice_thickness(src_ds)
    if st is not None:
        ds.SliceThickness = st

    # Window from per-frame (0028,9132) of first frame
    try:
        wl = src_ds.PerFrameFunctionalGroupsSequence[0][0x0028, 0x9132][0]
        ds.WindowCenter = str(wl[0x0028, 0x1050].value)
        ds.WindowWidth  = str(wl[0x0028, 0x1051].value)
    except Exception:
        pass

    # Rescale from per-frame (0028,9145) of first frame
    try:
        rs = src_ds.PerFrameFunctionalGroupsSequence[0][0x0028, 0x9145][0]
        ds.RescaleIntercept = str(rs[0x0028, 0x1052].value)
        ds.RescaleSlope     = str(rs[0x0028, 0x1053].value)
        ds.RescaleType      = str(rs[0x0028, 0x1054].value)
    except Exception:
        pass

    # Temporal
    ds.TriggerTime                = cardiac_delay_ms
    ds.TemporalPositionIdentifier = temporal_pos
    ds.NumberOfTemporalPositions  = n_phases

    # Remove Enhanced MR artefacts (in case any were copied via COPY_TAGS)
    for tag in REMOVE_TAGS:
        if tag in ds:
            del ds[tag]

    # Pixel data
    if pixel_2d is not None:
        if pixel_2d.dtype != np.uint16:
            pixel_2d = pixel_2d.astype(np.uint16)
        ds.PixelData = pixel_2d.tobytes()
        ds[0x7fe0, 0x0010].VR = 'OW'
    else:
        ds.PixelData = b''

    return ds


def convert_series(src_folder: str, dst_folder: str,
                   progress_cb, log_cb) -> None:
    """
    Main conversion pipeline.
    progress_cb(done, total) — called after each frame is written.
    log_cb(message)          — called for status messages.
    """
    os.makedirs(dst_folder, exist_ok=True)

    log_cb('Loading Enhanced MR files...')
    frames, src_ds = load_enhanced_mr_files(src_folder, log_cb)

    if not frames:
        log_cb('ERROR: No frames extracted.')
        return

    # Determine dimensions
    n_slices = max(f['in_stack_pos'] for f in frames)
    n_phases = max(f['temporal_pos']  for f in frames)
    total    = len(frames)
    log_cb(f'Extracted {total} frames: {n_slices} slices × {n_phases} phases')

    # One new SeriesInstanceUID shared by all output files
    series_uid = generate_uid(REALIZE_UID_PREFIX)
    log_cb(f'New SeriesInstanceUID: {series_uid}')

    # IOP once (shared across all frames)
    iop = extract_iop(src_ds)
    if iop:
        log_cb(f'ImageOrientationPatient: {iop}')
    else:
        log_cb('WARNING: ImageOrientationPatient not found; SliceLocation will be omitted.')

    log_cb('Writing Classic MR files...')
    for idx, frame in enumerate(frames):
        in_stack_pos = frame['in_stack_pos']
        temporal_pos = frame['temporal_pos']

        out_ds = _build_output_dataset(
            frame, src_ds, series_uid, n_slices, n_phases, iop
        )

        fname = f'slice{in_stack_pos:04d}_phase{temporal_pos:04d}.dcm'
        fpath = os.path.join(dst_folder, fname)
        out_ds.save_as(fpath, write_like_original=False)

        progress_cb(idx + 1, total)

    log_cb(f'Done. Wrote {total} files to: {dst_folder}')


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ConverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Enhanced MR → Classic DICOM Converter')
        self.resizable(True, True)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # --- Source row ---
        tk.Label(self, text='Source folder:').grid(row=0, column=0, sticky='e', **pad)
        self.src_var = tk.StringVar()
        self.src_entry = tk.Entry(self, textvariable=self.src_var, width=50)
        self.src_entry.grid(row=0, column=1, sticky='ew', **pad)
        tk.Button(self, text='Browse', command=self._browse_src).grid(row=0, column=2, **pad)

        # --- Detected info ---
        self.info_var = tk.StringVar(value='Detected: —')
        tk.Label(self, textvariable=self.info_var, fg='#555').grid(
            row=1, column=1, sticky='w', **pad)

        # --- Dest row ---
        tk.Label(self, text='Dest folder:').grid(row=2, column=0, sticky='e', **pad)
        self.dst_var = tk.StringVar()
        self.dst_entry = tk.Entry(self, textvariable=self.dst_var, width=50)
        self.dst_entry.grid(row=2, column=1, sticky='ew', **pad)
        tk.Button(self, text='Browse', command=self._browse_dst).grid(row=2, column=2, **pad)

        # --- Convert button ---
        self.convert_btn = tk.Button(
            self, text='Convert', width=16,
            command=self._start_convert, bg='#1a73e8', fg='white',
            font=('TkDefaultFont', 11, 'bold')
        )
        self.convert_btn.grid(row=3, column=0, columnspan=3, pady=10)

        # --- Progress bar ---
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value='')
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100.0, length=480
        )
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky='ew', padx=8, pady=2)
        tk.Label(self, textvariable=self.progress_label_var).grid(
            row=4, column=2, sticky='w', padx=4)

        # --- Log area ---
        self.log_text = scrolledtext.ScrolledText(
            self, height=14, width=72, state='disabled',
            font=('Courier', 9)
        )
        self.log_text.grid(row=5, column=0, columnspan=3, padx=8, pady=6, sticky='nsew')

        self.columnconfigure(1, weight=1)
        self.rowconfigure(5, weight=1)

    # ------------------------------------------------------------------
    # Browse callbacks (run in main thread)
    # ------------------------------------------------------------------
    def _browse_src(self):
        folder = filedialog.askdirectory(title='Select source folder (Enhanced MR multi-frame)')
        if folder:
            self.src_var.set(folder)
            self._scan_source(folder)

    def _browse_dst(self):
        folder = filedialog.askdirectory(title='Select destination folder')
        if folder:
            self.dst_var.set(folder)

    def _scan_source(self, folder):
        self.info_var.set('Scanning…')
        def _worker():
            result = scan_source_folder(folder)
            def _update():
                if result['valid']:
                    self.info_var.set(
                        f"Detected: {result['n_files']} files · "
                        f"{result['n_frames']} frames/file · "
                        f"{result['total']} total"
                    )
                    # Auto-suggest dest folder
                    if not self.dst_var.get():
                        parent = os.path.dirname(folder)
                        name   = os.path.basename(folder.rstrip('/\\'))
                        self.dst_var.set(os.path.join(parent, name + '_classic'))
                else:
                    self.info_var.set(f'Error: {result["error"]}')
            self.after(0, _update)
        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def _start_convert(self):
        src = self.src_var.get().strip()
        dst = self.dst_var.get().strip()

        if not src:
            self.log('ERROR: Please select a source folder.')
            return
        if not dst:
            self.log('ERROR: Please select a destination folder.')
            return
        if not os.path.isdir(src):
            self.log(f'ERROR: Source folder does not exist: {src}')
            return

        self.convert_btn.config(state='disabled')
        self.set_progress(0, 1)
        self.log(f'Starting conversion\n  src: {src}\n  dst: {dst}')

        def _worker():
            try:
                convert_series(
                    src, dst,
                    progress_cb=lambda done, total: self.after(
                        0, lambda: self.set_progress(done, total)
                    ),
                    log_cb=lambda msg: self.after(0, lambda m=msg: self.log(m)),
                )
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                self.after(0, lambda: self.log(f'EXCEPTION:\n{msg}'))
            finally:
                self.after(0, lambda: self.convert_btn.config(state='normal'))

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Thread-safe helpers (must be called via self.after() from workers)
    # ------------------------------------------------------------------
    def log(self, message: str):
        self.log_text.config(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def set_progress(self, done: int, total: int):
        pct = (done / total * 100) if total > 0 else 0
        self.progress_var.set(pct)
        self.progress_label_var.set(f'{done} / {total}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app = ConverterApp()
    app.mainloop()
