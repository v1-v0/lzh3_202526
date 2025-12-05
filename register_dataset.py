#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register_dataset.py

Register brightfield (BF) and fluorescence (FLUO) image pairs together with
Leica XML metadata for each dataset.

Assumed project layout (works on Windows / Linux / macOS):

  project_root/
    register_dataset.py      <-- this file
    source/
      10/
        10 P 1_ch00.tif      (BF)
        10 P 1_ch01.tif      (FLUO)
        10 P 2_ch00.tif
        10 P 2_ch01.tif
        ...
        MetaData/
          10 P 1.xml
          10 P 1_Properties.xml
          10 P 2.xml
          10 P 2_Properties.xml
          ...

Usage
-----

From Python:

    from pathlib import Path
    from register_dataset import register_all_datasets

    root = Path.cwd() / "source"
    records = register_all_datasets(root)

From command line:

    python register_dataset.py

This will scan ./source and print a short summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TypedDict

import re
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class CoreMeta(TypedDict):
    pixel_size_um: float
    field_length_um_x: float
    field_length_um_y: float


class PropertiesMeta(TypedDict, total=False):
    objective_name: str | None
    numerical_aperture: float | None
    optical_res_xy_um: float | None
    exposure_bf_s: float | None
    exposure_fluo_s: float | None
    # Leica viewer scaling
    bit_depth: int | None
    channel_scaling: list[dict]  # [{'black_norm': float, 'white_norm': float, 'gamma': float}, ...]


@dataclass
class ImagePairRecord:
    dataset: str
    pair_name: str          # e.g. "10 P 1"
    pair_index: int         # e.g. 1

    bf_path: Path           # BF image (*.tif, ch00)
    fluo_path: Path         # FLUO image (*.tif, ch01 or variant)

    # Metadata paths
    meta_core: Optional[Path]       # "<name>.xml"
    meta_props: Optional[Path]      # "<name>_Properties.xml"

    # Parsed numeric metadata
    pixel_size_um: Optional[float]
    field_length_um_x: Optional[float]
    field_length_um_y: Optional[float]
    objective_name: Optional[str]
    numerical_aperture: Optional[float]
    optical_res_xy_um: Optional[float]
    exposure_bf_s: Optional[float]
    exposure_fluo_s: Optional[float]

    # Leica viewer scaling metadata
    bit_depth: Optional[int] = None
    channel_scaling: Optional[list[dict]] = None


# ---------------------------------------------------------------------------
# XML parsers
# ---------------------------------------------------------------------------

def parse_core_xml(path: Path) -> Optional[CoreMeta]:
    """
    Parse Leica '<name>.xml' to get pixel size and field lengths.

    The compact XML encodes e.g.:

      <Dimensions>
        <DimensionDescription DimID="1"
                              NumberOfElements="1200"
                              Length="1.313922e-004"
                              Unit="m" ... />
        <DimensionDescription DimID="2"
                              NumberOfElements="1200"
                              Length="1.313922e-004"
                              Unit="m" ... />

    Returns a dict with:
      pixel_size_um
      field_length_um_x
      field_length_um_y

    Returns None on error or missing information.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return None

    dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
    if len(dims) < 2:
        print(f"[WARN] No dimension info in {path}")
        return None

    dim_x, dim_y = dims[0], dims[1]

    try:
        n_x = int(dim_x.attrib["NumberOfElements"])
        n_y = int(dim_y.attrib["NumberOfElements"])
        length_x_m = float(dim_x.attrib["Length"])
        length_y_m = float(dim_y.attrib["Length"])
    except (KeyError, ValueError) as e:
        print(f"[WARN] Problem with dimension attributes in {path}: {e}")
        return None

    pixel_size_x_um = (length_x_m * 1e6) / n_x
    pixel_size_y_um = (length_y_m * 1e6) / n_y

    core_info: CoreMeta = {
        "pixel_size_um": (pixel_size_x_um + pixel_size_y_um) / 2.0,
        "field_length_um_x": length_x_m * 1e6,
        "field_length_um_y": length_y_m * 1e6,
    }

    return core_info


def parse_properties_xml(path: Path) -> PropertiesMeta:
    """
    Parse '<name>_Properties.xml' for:
      - objective name
      - numerical aperture
      - optical resolution XY
      - exposure times for BF and FLUO
      - bit depth
      - Leica viewer scaling (per-channel black/white/gamma in normalized units)

    Returns a dict with keys:
      objective_name, numerical_aperture,
      optical_res_xy_um, exposure_bf_s, exposure_fluo_s,
      bit_depth, channel_scaling

    Missing values are returned as None or empty list.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return {}

    img = root.find(".//Image")
    if img is None:
        return {}

    # Objective / NA ---------------------------------------------------------
    atld = img.find(".//ATLCameraSettingDefinition")
    objective_name: Optional[str] = None
    numerical_aperture: Optional[float] = None

    if atld is not None:
        objective_name = atld.attrib.get("ObjectiveName")
        na_raw = atld.attrib.get("NumericalAperture")
        try:
            numerical_aperture = float(na_raw) if na_raw is not None else None
        except ValueError:
            numerical_aperture = None

    # Optical resolution XY (first channel) ----------------------------------
    optical_res_xy_um: Optional[float] = None
    ch = img.find(".//ImageDescription/Channels/ChannelDescription")
    if ch is not None:
        xy_str = ch.attrib.get("OpticalResolutionXY")
        if xy_str:
            token = xy_str.split()[0]
            try:
                optical_res_xy_um = float(token)
            except ValueError:
                optical_res_xy_um = None

    # Bit depth from ChannelDescription Resolution ---------------------------
    bit_depth: Optional[int] = None
    if ch is not None:
        res = ch.attrib.get("Resolution")
        if res is not None:
            try:
                bit_depth = int(res)
            except ValueError:
                bit_depth = None

    # Exposure times from WideFieldChannelInfo -------------------------------
    exposure_bf_s: Optional[float] = None
    exposure_fluo_s: Optional[float] = None

    for wfi in img.findall(".//WideFieldChannelConfigurator/WideFieldChannelInfo"):
        method = wfi.attrib.get("ContrastingMethodName", "")
        exp = wfi.attrib.get("ExposureTime")
        if not exp:
            continue

        token = exp.split()[0]
        try:
            val = float(token)
            # Heuristic: values > 10 are probably in ms, otherwise seconds
            if val > 10:
                exp_s = val / 1000.0
            else:
                exp_s = val
        except ValueError:
            exp_s = None

        if method == "TL-BF":
            exposure_bf_s = exp_s
        elif method == "FLUO" and exposure_fluo_s is None:
            exposure_fluo_s = exp_s

    # Leica viewer scaling (ChannelScalingInfo) ------------------------------
    channel_scaling: list[dict] = []
    for csi in img.findall(".//Attachment[@Name='ViewerScaling']//ChannelScalingInfo"):
        try:
            white = float(csi.attrib.get("WhiteValue", ""))
            black = float(csi.attrib.get("BlackValue", ""))
            gamma = float(csi.attrib.get("GammaValue", "1"))
        except (TypeError, ValueError):
            continue

        channel_scaling.append(
            {
                "black_norm": black,
                "white_norm": white,
                "gamma": gamma,
            }
        )

    return {
        "objective_name": objective_name,
        "numerical_aperture": numerical_aperture,
        "optical_res_xy_um": optical_res_xy_um,
        "exposure_bf_s": exposure_bf_s,
        "exposure_fluo_s": exposure_fluo_s,
        "bit_depth": bit_depth,
        "channel_scaling": channel_scaling,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_dataset(folder: str | Path) -> List[ImagePairRecord]:
    """
    Scan a single dataset folder (e.g. 'source/10', 'source/11', 'source/16', 'source/19')
    and register BF, FLUO and metadata files.
    """
    folder = Path(folder)
    dataset_name = folder.name
    meta_dir = folder / "MetaData"

    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    # All BF files end with _ch00.tif
    bf_files = sorted(
        
        f for f in folder.glob("*_ch00.tif")
        if not f.name.startswith("._")
    )
    
    # Pattern:
    #   ^(.*)\s(\d+)_ch00\.tif$
    pattern = re.compile(r"^(.*)\s(\d+)_ch00\.tif$")

    records: List[ImagePairRecord] = []

    for bf in bf_files:
        m = pattern.match(bf.name)
        if not m:
            # Not a recognized BF naming pattern; skip it
            continue

        prefix = m.group(1)
        pair_index = int(m.group(2))

        # pair_name is simply "<prefix> <index>" i.e. base name without _ch00.tif
        pair_name = f"{prefix} {pair_index}"

        # Find FLUO file for this pair_name (e.g. '<pair_name>_ch01.tif')
        fluo = _find_fluo_file(folder, pair_name)
        if fluo is None:
            print(f"[WARN] No FLUO file found for BF {bf.name}")
            continue

        # Metadata paths (if present)
        meta_core: Optional[Path] = None
        meta_props: Optional[Path] = None

        if meta_dir.is_dir():
            core_candidate = meta_dir / f"{pair_name}.xml"
            props_candidate = meta_dir / f"{pair_name}_Properties.xml"
            if core_candidate.exists():
                meta_core = core_candidate
            if props_candidate.exists():
                meta_props = props_candidate

        if meta_core is None and meta_props is None:
            print(f"[WARN] No metadata XML for '{pair_name}' in {meta_dir}")

        # Defaults
        pixel_size_um: Optional[float] = None
        field_len_x_um: Optional[float] = None
        field_len_y_um: Optional[float] = None
        objective_name: Optional[str] = None
        numerical_aperture: Optional[float] = None
        optical_res_xy_um: Optional[float] = None
        exposure_bf_s: Optional[float] = None
        exposure_fluo_s: Optional[float] = None
        bit_depth: Optional[int] = None
        channel_scaling: Optional[list[dict]] = None

        # Geometry from core XML
        if meta_core is not None:
            core_info = parse_core_xml(meta_core)
            if core_info is not None:
                pixel_size_um = core_info["pixel_size_um"]
                field_len_x_um = core_info["field_length_um_x"]
                field_len_y_um = core_info["field_length_um_y"]

        # Extra info from _Properties XML
        if meta_props is not None:
            props_info = parse_properties_xml(meta_props)
            objective_name = props_info.get("objective_name")
            numerical_aperture = props_info.get("numerical_aperture")
            optical_res_xy_um = props_info.get("optical_res_xy_um")
            exposure_bf_s = props_info.get("exposure_bf_s")
            exposure_fluo_s = props_info.get("exposure_fluo_s")
            bit_depth = props_info.get("bit_depth")
            channel_scaling = props_info.get("channel_scaling")

        records.append(
            ImagePairRecord(
                dataset=dataset_name,
                pair_name=pair_name,
                pair_index=pair_index,
                bf_path=bf,
                fluo_path=fluo,
                meta_core=meta_core,
                meta_props=meta_props,
                pixel_size_um=pixel_size_um,
                field_length_um_x=field_len_x_um,
                field_length_um_y=field_len_y_um,
                objective_name=objective_name,
                numerical_aperture=numerical_aperture,
                optical_res_xy_um=optical_res_xy_um,
                exposure_bf_s=exposure_bf_s,
                exposure_fluo_s=exposure_fluo_s,
                bit_depth=bit_depth,
                channel_scaling=channel_scaling,
            )
        )

    return records


def _find_fluo_file(folder: Path, pair_name: str) -> Optional[Path]:
    """
    Try to locate the FLUO image for a given pair_name in 'folder'.

    Primary pattern: '<pair_name>_ch01.tif'

    Fallback patterns can be extended if your naming varies.
    """
    fluo = folder / f"{pair_name}_ch01.tif"
    if fluo.exists():
        return fluo

    # Fallbacks: adjust here if your naming is different
    candidates = []
    candidates.extend(folder.glob(f"{pair_name}_ch1.tif"))
    candidates.extend(folder.glob(f"{pair_name}_ch01.tiff"))
    candidates.extend(folder.glob(f"{pair_name}_ch1.tiff"))

    if candidates:
        chosen = candidates[0]
        print(f"[INFO] Using alternative FLUO file for {pair_name}: {chosen.name}")
        return chosen

    return None


def register_single_dataset(source_root: str | Path, dataset_id: str | int) -> List[ImagePairRecord]:
    """
    Convenience wrapper: register exactly one dataset folder under source_root.
    """
    source_root = Path(source_root)
    ds_folder = source_root / str(dataset_id)
    return register_dataset(ds_folder)


def register_all_datasets(source_root: str | Path) -> List[ImagePairRecord]:
    """
    Scan all subfolders under 'source_root' and return a flat list
    of ImagePairRecord.
    """
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    all_records: List[ImagePairRecord] = []

    for ds_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        all_records.extend(register_dataset(ds_dir))

    return all_records


# ---------------------------------------------------------------------------
# Small command-line demo
# ---------------------------------------------------------------------------

def _check_registered_files(records: List[ImagePairRecord]) -> None:
    """
    Print any missing BF/FLUO files referenced by the registry.
    In normal use there should be none.
    """
    missing = []
    for r in records:
        if not r.bf_path.exists():
            missing.append(("BF", r.pair_name, r.bf_path))
        if not r.fluo_path.exists():
            missing.append(("FLUO", r.pair_name, r.fluo_path))

    if not missing:
        print("All BF/FLUO image paths exist on disk.")
    else:
        print("Missing files:")
        for kind, name, path in missing:
            print(f"  [{kind}] {name}: {path}")


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"

    print("Project root :", project_root)
    print("Source root  :", source_root)

    records = register_all_datasets(source_root)
    print(f"\nTotal pairs found: {len(records)}")

    for r in records[:5]:
        print("\nPair:", r.pair_name)
        print("  BF:", r.bf_path)
        print("  FLUO:", r.fluo_path)
        print("  pixel_size_um:", r.pixel_size_um)
        print("  field_x_um   :", r.field_length_um_x)
        print("  objective    :", r.objective_name)
        print("  NA           :", r.numerical_aperture)
        print("  exposure BF/FLUO (s):", r.exposure_bf_s, r.exposure_fluo_s)
        print("  bit_depth    :", r.bit_depth)
        print("  channel_scaling:", r.channel_scaling)

    print()
    _check_registered_files(records)