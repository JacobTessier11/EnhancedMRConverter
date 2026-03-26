# Enhanced MR → Classic DICOM Converter

A small GUI tool that converts **Enhanced MR Image Storage** (multi-frame DICOM) into **Classic MR Image Storage** (single-frame DICOM), enabling 4D cardiac datasets to be loaded in software packages such as [3D Slicer](https://www.slicer.org/), OsiriX, Horos, and other DICOM viewers that expect one file per frame.

---

## Background

Modern MRI scanners (Siemens, Philips, GE) increasingly store 4D acquisitions — multiple slices across multiple cardiac phases — as **Enhanced MR** (SOP Class `1.2.840.10008.5.1.4.1.1.4.1`). In this format, all frames for an entire series are packed into a small number of multi-frame files, with spatial and temporal metadata stored in per-frame functional groups rather than at the top level of each file.

Many analysis and visualization packages still expect the older **Classic MR** format (SOP Class `1.2.840.10008.5.1.4.1.1.4`), where each slice/phase combination is a separate `.dcm` file with flat top-level tags like `SliceLocation`, `TriggerTime`, `ImageOrientationPatient`, and `PixelSpacing`. This mismatch causes import failures or incorrect sorting when working with 4D cardiac data.

This tool bridges that gap.

---

## What it does

- Reads all Enhanced MR `.dcm` files in a source folder
- Extracts every frame with its spatial position, orientation, cardiac trigger delay, and pixel data
- Writes one Classic MR `.dcm` file per frame, named `slice####_phase####.dcm`
- Correctly propagates tags that Siemens scanners store exclusively in per-frame functional groups:
  - `ImageOrientationPatient`
  - `PixelSpacing` and `SliceThickness`
  - `WindowCenter` / `WindowWidth`
  - `RescaleIntercept` / `RescaleSlope` / `RescaleType`
  - `CardiacTriggerDelayTime` → `TriggerTime`
- Generates a new shared `SeriesInstanceUID` for the output series
- Computes `SliceLocation` from `ImagePositionPatient` and the slice normal

The result can be dropped directly into 3D Slicer's DICOM browser and will be detected as a multi-phase cardiac volume.

---

## Requirements

- Python 3.9+
- [pydicom](https://pydicom.github.io/) — `pip install pydicom`
- [numpy](https://numpy.org/) — `pip install numpy`
- tkinter (included with standard Python on Windows and macOS)

---

## Usage

### Pre-built executable (Windows)

Download `EnhancedMR_Converter.exe` from the [Releases](../../releases) page. No Python installation required. Double-click to launch.

### From source

```bash
pip install pydicom numpy
python enhanced_mr_converter.py
```

### Steps

1. Click **Browse** next to *Source folder* and select the folder containing your Enhanced MR `.dcm` files.
2. The tool will detect the file count and frames-per-file automatically.
3. A destination folder is suggested automatically (source name + `_classic`). Override if needed.
4. Click **Convert**. Progress is shown in the log area.
5. Open the output folder in your DICOM viewer.

---

## Input requirements

| Property | Expected |
|---|---|
| SOP Class | `1.2.840.10008.5.1.4.1.1.4.1` (Enhanced MR Image Storage) |
| Pixel data | Any transfer syntax supported by pydicom |
| Temporal index | `TemporalPositionIndex` in `FrameContentSequence` |
| Slice index | `InStackPositionNumber` in `FrameContentSequence` |

Tested with Siemens MAGNETOM 4D Flow acquisitions. Should work with any compliant Enhanced MR series.

---

## Building the executable

The repo includes `EnhancedMR_Converter.spec` for reproducible PyInstaller builds:

```bash
pip install pyinstaller pydicom numpy
pyinstaller EnhancedMR_Converter.spec
```

The output will be at `dist/EnhancedMR_Converter.exe`.

---

## License

MIT
