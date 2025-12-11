# metadata.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


@dataclass
class ChannelMeta:
    name: str
    channel_id: int
    kind: str              # e.g. "TL-BF", "FLUO"
    exposure: float
    emission_nm: Optional[float]


@dataclass
class ImageMeta:
    batch_id: int
    image_name: str            # base_id, not including _ch00/_ch01
    width_px: int
    height_px: int
    pixel_size_m: float
    objective: str
    numerical_aperture: float
    channels: Dict[str, ChannelMeta]
    time_stamps_raw: List[str]


def parse_leica_xml(xml_path: Path, batch_id: int, image_name: str) -> ImageMeta:
    """
    Parse Leica LAS AF / LAS X XML file to extract key metadata.

    NOTE: This assumes a structure similar to what we discussed. You may need
    to tweak XPath expressions based on your actual XML schema.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ---- Dimensions and pixel size ----
    # Example structure:
    # <ImageDescription>
    #   <Dimensions>
    #       <DimensionDescription DimID="1" NumberOfElements="2048" Length="2.048E-3" .../>
    #       <DimensionDescription DimID="2" .../>
    #   </Dimensions>
    # </ImageDescription>
    dims = root.find(".//ImageDescription/Dimensions")
    if dims is None:
        raise RuntimeError(f"Could not find Dimensions in XML: {xml_path}")

    dim_x = dims.find("./DimensionDescription[@DimID='1']")
    dim_y = dims.find("./DimensionDescription[@DimID='2']")

    if dim_x is None or dim_y is None:
        raise RuntimeError(f"Could not find DimensionDescription for X/Y in XML: {xml_path}")

    width_px = int(dim_x.attrib["NumberOfElements"])
    height_px = int(dim_y.attrib["NumberOfElements"])
    length_x_m = float(dim_x.attrib["Length"])
    # Assume square pixels: pixel size = Length / NumberOfElements
    pixel_size_m = length_x_m / width_px

    # ---- Hardware / objective / NA ----
    # Example structure might be:
    # <Attachment Name="HardwareSetting">
    #   <ATLCameraSettingDefinition ObjectiveName="40x" NumericalAperture="0.8" ... />
    #   <WideFieldChannelInfo ... />
    # </Attachment>
    hw = root.find(".//Attachment[@Name='HardwareSetting']")
    objective = ""
    na = 0.0

    if hw is not None:
        cam_def = hw.find(".//ATLCameraSettingDefinition")
        if cam_def is not None:
            objective = cam_def.attrib.get("ObjectiveName", "")
            try:
                na = float(cam_def.attrib.get("NumericalAperture", "0"))
            except ValueError:
                na = 0.0

    # ---- Channels ----
    # Typical structure:
    # <WideFieldChannelInfo Channel="1" ContrastingMethodName="TL-BF"
    #     UserDefName="Brightfield" ExposureTime="50" EmissionWavelength="520" ... />
    channels: Dict[str, ChannelMeta] = {}
    if hw is not None:
        ch_nodes = hw.findall(".//WideFieldChannelInfo")
        for ch in ch_nodes:
            try:
                channel_id = int(ch.attrib["Channel"])
            except (KeyError, ValueError):
                continue

            kind = ch.attrib.get("ContrastingMethodName", "")
            exposure_str = ch.attrib.get("ExposureTime", "0")
            try:
                exposure = float(exposure_str)
            except ValueError:
                exposure = 0.0

            emission_nm_str = ch.attrib.get("EmissionWavelength")
            if emission_nm_str is not None:
                try:
                    emission_nm = float(emission_nm_str)
                except ValueError:
                    emission_nm = None
            else:
                emission_nm = None

            user_name = ch.attrib.get("UserDefName", "")

            if kind == "TL-BF":
                key = "brightfield"
            elif kind == "FLUO":
                key = "fluorescence"
            else:
                # Ignore autofocus or other channels for now
                key = f"channel_{channel_id}"

            channels[key] = ChannelMeta(
                name=user_name or key,
                channel_id=channel_id,
                kind=kind,
                exposure=exposure,
                emission_nm=emission_nm,
            )

    # ---- Timestamps ----
    time_stamps_raw: List[str] = []
    ts_node = root.find(".//TimeStampList")
    if ts_node is not None and ts_node.text:
        # Often a whitespace-separated list of timestamps
        time_stamps_raw = ts_node.text.split()

    return ImageMeta(
        batch_id=batch_id,
        image_name=image_name,
        width_px=width_px,
        height_px=height_px,
        pixel_size_m=pixel_size_m,
        objective=objective,
        numerical_aperture=na,
        channels=channels,
        time_stamps_raw=time_stamps_raw,
    )


def image_meta_to_dict(meta: ImageMeta) -> Dict[str, Any]:
    """
    Convert ImageMeta to a plain dictionary for JSON saving.
    """
    d = asdict(meta)
    d["channels"] = {k: asdict(v) for k, v in meta.channels.items()}
    return d