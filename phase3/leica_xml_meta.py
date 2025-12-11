from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional


@dataclass
class LeicaMeta:
    # Basic
    image_name: str = ""
    start_time_raw: str = ""
    end_time_raw: str = ""

    # Geometry
    nx: int = 0
    ny: int = 0
    fov_x_um: float = 0.0
    fov_y_um: float = 0.0
    voxel_um: float = 0.0

    # Optical / camera
    objective_name: str = ""
    magnification: float = 0.0          # nominal mag (e.g. 100x)
    total_video_mag: float = 0.0
    camera_name: str = ""
    camera_name_pure: str = ""
    theo_sensor_px_m: float = 0.0       # TheoCamSensorPixelSizeX
    eff_px_um: float = 0.0             # effective pixel size in µm

    # Channels
    n_channels: int = 0
    bf_channel_name: str = ""
    bf_contrast: str = ""
    bf_cube: str = ""
    fluo_channel_name: str = ""
    fluo_contrast: str = ""
    fluo_cube: str = ""

    # Exposure (assumed same for both channels; ms)
    exposure_ms: Optional[float] = None

    # Extra
    numerical_aperture: Optional[float] = None
    immersion: Optional[str] = None
    stage_pos_x: Optional[str] = None
    stage_pos_y: Optional[str] = None
    target_temp_c: Optional[float] = None
    current_temp_c: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# Internal XML helpers
# ---------------------------------------------------------------------


def _get_first(elem: ET.Element, path: str) -> Optional[ET.Element]:
    return elem.find(path)


def _parse_float(text: Optional[str], default: Optional[float] = 0.0) -> Optional[float]:
    """
    Safe float parser that can return None if default is None.
    """
    if text is None:
        return default
    try:
        return float(text)
    except Exception:
        return default


def _parse_int(text: Optional[str], default: int = 0) -> int:
    if text is None:
        return default
    try:
        return int(text)
    except Exception:
        return default


def _strip_ms(time_str: str) -> str:
    # "10/10/2025 15:57:04.342" -> "10/10/2025 15:57:04"
    if "." in time_str:
        return time_str.split(".", 1)[0]
    return time_str


# ---------------------------------------------------------------------
# Main parsing entry point
# ---------------------------------------------------------------------


def load_leica_metadata(
    properties_xml_path: Path,
    main_xml_path: Optional[Path] = None,
) -> LeicaMeta:
    """
    Parse Leica LAS X XML metadata from:
    - properties_xml_path: the *Properties.xml file (rich per-image metadata)
    - main_xml_path: the base .xml for the same image (hardware details)
      If None, we will try to guess (same name without '_Properties').

    Returns a LeicaMeta object.
    """
    properties_xml_path = Path(properties_xml_path)
    if main_xml_path is None:
        # Guess: "18 N NO 1_Properties.xml" -> "18 N NO 1.xml"
        if properties_xml_path.name.endswith("_Properties.xml"):
            base = properties_xml_path.name.removesuffix("_Properties.xml")
            main_xml_path = properties_xml_path.with_name(base + ".xml")
        else:
            main_xml_path = None

    meta = LeicaMeta()

    # --------------------------------------------------------------
    # Parse Properties.xml (geometry, image name, time, channels)
    # --------------------------------------------------------------
    if properties_xml_path.is_file():
        try:
            tree = ET.parse(properties_xml_path)
            root = tree.getroot()
        except Exception:
            root = None
    else:
        root = None

    if root is not None:
        image_elem = _get_first(root, "Image")
        if image_elem is None:
            # Some variants may omit <Image> wrapper
            image_elem = root

        img_desc = _get_first(image_elem, "ImageDescription")
        if img_desc is not None:
            # Image name
            name_elem = _get_first(img_desc, "Name")
            if name_elem is not None and name_elem.text:
                meta.image_name = name_elem.text.strip()

            # Number of channels
            nch_elem = _get_first(img_desc, "NumberOfChannels")
            if nch_elem is not None and nch_elem.text:
                meta.n_channels = _parse_int(nch_elem.text, 0)

            # Dimensions
            dims = _get_first(img_desc, "Dimensions")
            if dims is not None:
                xs = [
                    d for d in dims.findall("DimensionDescription")
                    if d.get("DimID") in ("X", "1")
                ]
                ys = [
                    d for d in dims.findall("DimensionDescription")
                    if d.get("DimID") in ("Y", "2")
                ]
                if xs:
                    dx = xs[0]
                    meta.nx = _parse_int(dx.get("NumberOfElements"), 0)
                    meta.fov_x_um = _parse_float(dx.get("Length"), 0.0) or 0.0
                    meta.voxel_um = _parse_float(dx.get("Voxel"), 0.0) or 0.0
                if ys:
                    dy = ys[0]
                    meta.ny = _parse_int(dy.get("NumberOfElements"), 0)
                    meta.fov_y_um = _parse_float(dy.get("Length"), 0.0) or 0.0

            # Start/end time
            st = _get_first(img_desc, "StartTime")
            et = _get_first(img_desc, "EndTime")
            if st is not None and st.text:
                meta.start_time_raw = st.text.strip()
            if et is not None and et.text:
                meta.end_time_raw = et.text.strip()

            # FLUO description (channel names / cubes)
            fluo_desc = _get_first(img_desc, "FluoDescription")
            if fluo_desc is not None:
                recs = fluo_desc.findall("FluoDescriptionRecord")
                for rec in recs:
                    ident = rec.get("Identifier", "").strip()
                    name = rec.get("Name", "").strip()
                    contrast = rec.get("Contrast", "").strip()
                    cube = rec.get("Cube", "").strip()
                    if ident == "Channel 0":
                        meta.bf_channel_name = name
                        meta.bf_contrast = contrast
                        meta.bf_cube = cube
                    elif ident == "Channel 1":
                        meta.fluo_channel_name = name
                        meta.fluo_contrast = contrast
                        meta.fluo_cube = cube

    # --------------------------------------------------------------
    # Parse main XML (hardware / camera / objective / pixel size)
    # --------------------------------------------------------------
    cam_root = None
    if main_xml_path is not None and Path(main_xml_path).is_file():
        try:
            cam_tree = ET.parse(main_xml_path)
            cam_root = cam_tree.getroot()
        except Exception:
            cam_root = None

    if cam_root is not None:
        image_elem2 = _get_first(cam_root, "Image")
        if image_elem2 is None:
            image_elem2 = cam_root

        # HardwareSetting
        for attach in image_elem2.findall("Attachment"):
            if attach.get("Name") == "HardwareSetting":
                hw = attach.find("ATLCameraSettingDefinition")
                if hw is None:
                    continue

                # Objective
                meta.objective_name = hw.get("ObjectiveName", meta.objective_name).strip()
                meta.numerical_aperture = _parse_float(
                    hw.get("NumericalAperture"), None
                )
                meta.immersion = hw.get("Immersion")
                meta.stage_pos_x = hw.get("StagePosX")
                meta.stage_pos_y = hw.get("StagePosY")
                meta.target_temp_c = _parse_float(hw.get("TargetTemperature"), None)
                meta.current_temp_c = _parse_float(hw.get("CurrentTemperature"), None)

                # Magnification
                meta.magnification = _parse_float(hw.get("Magnification"), 0.0) or 0.0
                meta.total_video_mag = _parse_float(hw.get("TotalVideoMag"), 0.0) or 0.0

                # Theoretical sensor pixel size
                meta.theo_sensor_px_m = _parse_float(
                    hw.get("TheoCamSensorPixelSizeX"), 0.0
                ) or 0.0

                # Camera / WideFieldChannelConfigurator
                wfc = hw.find("WideFieldChannelConfigurator")
                if wfc is not None:
                    meta.camera_name = wfc.get("CameraName", "").strip()
                    meta.camera_name_pure = wfc.get("CameraNamePure", "").strip()

                    # Exposure: take the first exposure time
                    for wch in wfc.findall("WideFieldChannelInfo"):
                        exp = wch.get("ExposureTime")
                        if exp:
                            etxt = exp.strip()
                            if etxt.endswith("ms"):
                                meta.exposure_ms = _parse_float(
                                    etxt.replace("ms", ""), 0.0
                                )
                            else:
                                secs = _parse_float(etxt, 0.0) or 0.0
                                meta.exposure_ms = secs * 1000.0
                            break

                break  # only first HardwareSetting

    # --------------------------------------------------------------
    # Derived values: effective pixel size from sensor + magnification
    # --------------------------------------------------------------
    if meta.theo_sensor_px_m > 0.0 and meta.magnification > 0:
        sensor_um = meta.theo_sensor_px_m * 1e6
        meta.eff_px_um = sensor_um / meta.magnification
    else:
        # Fallback: compute from FOV and pixels
        if meta.nx > 0 and meta.fov_x_um > 0:
            meta.eff_px_um = meta.fov_x_um / meta.nx

    return meta


# ---------------------------------------------------------------------
# Spec-sheet builder
# ---------------------------------------------------------------------


def build_spec_sheet(meta: LeicaMeta) -> str:
    """
    Build a CSV-like spec sheet:

    Parameter,Value from Properties file,Notes
    ...
    """
    m = meta.as_dict()
    lines: list[str] = []
    add = lines.append

    add("Parameter,Value from Properties file,Notes")

    # Image name
    add(f"Image name,{m['image_name']},")

    # Acquisition date/time
    if m["start_time_raw"] and m["end_time_raw"]:
        st_short = _strip_ms(m["start_time_raw"])
        et_short = _strip_ms(m["end_time_raw"])
        add(f"Acquisition date/time,{st_short} → {et_short},")
    elif m["start_time_raw"]:
        add(f"Acquisition date/time,{_strip_ms(m['start_time_raw'])},")
    else:
        add("Acquisition date/time,,")

    # Objective & magnification
    add(f"Objective,{m['objective_name']},")
    total_mag: float = m["total_video_mag"] if m["total_video_mag"] else m["magnification"]
    if total_mag:
        add(f"Total magnification,{int(total_mag)}×,")
    else:
        add("Total magnification,,")

    # Camera
    cam_full: str = m["camera_name"]
    cam_pure: str = m["camera_name_pure"]
    if cam_full and cam_pure:
        cam_val = f"{cam_full} ({cam_pure})"
    else:
        cam_val = cam_full or cam_pure
    add(f"Camera,{cam_val},")

    # Sensor & effective pixel size
    if m["theo_sensor_px_m"] > 0:
        sensor_um = m["theo_sensor_px_m"] * 1e6
        add(
            "Sensor pixel size,"
            f"{sensor_um:.0f} µm,"
            f"TheoCamSensorPixelSizeX/Y = {m['theo_sensor_px_m']} m"
        )
    else:
        add("Sensor pixel size,,No TheoCamSensorPixelSize in metadata")

    eff_px_um: float = m["eff_px_um"] or 0.0
    if eff_px_um > 0:
        add(
            "Effective pixel size (exact),"
            f"{eff_px_um:.3f} µm/pixel,"
            "= sensor pixel / total magnification (or FOV / pixels)"
        )
    else:
        add("Effective pixel size (exact),,")

    # Geometry
    nx: int = m["nx"]
    ny: int = m["ny"]
    fov_x: float = m["fov_x_um"]
    fov_y: float = m["fov_y_um"]
    add(
        "FOV X/Y,"
        f"{fov_x:.2f} µm × {fov_y:.2f} µm,"
        "Length in metadata"
    )
    add(f"Pixels X/Y,{nx} × {ny},")

    if nx > 0 and fov_x > 0:
        calc_px = fov_x / nx
        add(
            "Calculated pixel size,"
            f"{fov_x:.2f} / {nx} = {calc_px:.7f} µm/pixel,"
            f"Check vs effective {eff_px_um:.3f} – use {eff_px_um:.3f} in code"
        )
    else:
        add("Calculated pixel size,,")

    if m["voxel_um"] > 0:
        add(
            "Voxel (explicit),"
            f"{m['voxel_um']:.3f} µm/pixel,"
            "Rounded value stored by Leica"
        )
    else:
        add("Voxel (explicit),,")

    # Channels summary
    bf_contrast = m["bf_contrast"] or "TL-BF"
    fluo_contrast = m["fluo_contrast"] or "FLUO"
    n_ch: int = m["n_channels"] or 0
    ch_summary = f"{bf_contrast} + {fluo_contrast}"
    add(
        "Channels,"
        f"{n_ch} ({ch_summary}),"
        f"Cube {m['bf_cube']} for BF, cube {m['fluo_cube']} for FLUO"
    )

    # Extra (optional)
    if m["numerical_aperture"] is not None:
        add(f"Numerical aperture,{m['numerical_aperture']},")

    if m["immersion"]:
        add(f"Immersion,{m['immersion']},")

    if m["exposure_ms"] is not None:
        add(f"Exposure time,{m['exposure_ms']:.0f} ms,")

    if m["stage_pos_x"] or m["stage_pos_y"]:
        add(
            "Stage position,"
            f"{m['stage_pos_x']} , {m['stage_pos_y']},"
        )

    if m["target_temp_c"] is not None and m["current_temp_c"] is not None:
        add(
            "Camera temperature,"
            f"Target {m['target_temp_c']} °C, current {m['current_temp_c']} °C,"
        )

    return "\n".join(lines)