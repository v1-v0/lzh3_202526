# metadata.py
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import xml.etree.ElementPath as ElementPath


@dataclass
class ChannelMeta:
    name: str
    channel_id: int
    kind: str              # "BF" or "FLUO"
    exposure: float
    emission_nm: float | None


@dataclass
class ImageMeta:
    batch_id: int
    image_name: str
    width_px: int
    height_px: int
    pixel_size_m: float
    objective: str
    numerical_aperture: float
    channels: Dict[str, ChannelMeta]
    time_stamps_raw: List[str]

def find(self, path, namespaces=None):
    """Find first matching element by tag name or path.

    *path* is a string having either an element tag or an XPath,
    *namespaces* is an optional mapping from namespace prefix to full name.

    Return the first matching element, or None if no element was found.

    """
    if self is None:
        return None  # Or raise a custom error for clarity
    return ElementPath.find(self, path, namespaces)

def parse_leica_xml(xml_path: Path, batch_id: int, image_name: str) -> ImageMeta:
    """
    Parse Leica LAS AF / LAS X XML like '11 N NO 1.xml' to extract key metadata.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Dimensions
    dims = root.find(".//ImageDescription/Dimensions")
    if dims is None:
        raise ValueError("Dimensions node not found in XML at .//ImageDescription/Dimensions")
    dim_x = dims.find("./DimensionDescription[@DimID='1']")
    dim_y = dims.find("./DimensionDescription[@DimID='2']")
    if dim_x is None or dim_y is None:
        raise ValueError("Required DimensionDescription nodes (DimID='1' or DimID='2') not found in XML")
    width_px = int(dim_x.attrib["NumberOfElements"])
    height_px = int(dim_y.attrib["NumberOfElements"])
    length_x_m = float(dim_x.attrib["Length"])
    # Assume square pixels, so:
    pixel_size_m = length_x_m / width_px

    # Hardware attachment
    hw = root.find(".//Attachment[@Name='HardwareSetting']")
    if hw is not None:
        cam_def = hw.find(".//ATLCameraSettingDefinition")
        if cam_def is not None:
            obj_name = cam_def.attrib.get("ObjectiveName", "")
            na = float(cam_def.attrib.get("NumericalAperture", "0"))
        else:
            obj_name = ""
            na = 0.0
    else:
        obj_name = ""
        na = 0.0

    # WideFieldChannelInfo nodes – identify BF vs FLUO
    ch_nodes = hw.findall(".//WideFieldChannelInfo") if hw is not None else []
    channels: Dict[str, ChannelMeta] = {}
    for ch in ch_nodes:
        channel_id = int(ch.attrib["Channel"])
        kind = ch.attrib.get("ContrastingMethodName", "")
        exposure = float(ch.attrib.get("ExposureTime", "0"))
        emission_nm = ch.attrib.get("EmissionWavelength")
        emission_nm = float(emission_nm) if emission_nm is not None else None

        if kind == "TL-BF":
            key = "brightfield"
        elif kind == "FLUO":
            key = "fluorescence"
        else:
            # skip autofocus or others for now
            continue

        channels[key] = ChannelMeta(
            name=ch.attrib.get("UserDefName", key),
            channel_id=channel_id,
            kind=kind,
            exposure=exposure,
            emission_nm=emission_nm
        )

    # Time stamps
    ts_node = root.find(".//TimeStampList")
    if ts_node is not None and ts_node.text:
        time_stamps_raw = ts_node.text.split()
    else:
        time_stamps_raw = []

    return ImageMeta(
        batch_id=batch_id,
        image_name=image_name,
        width_px=width_px,
        height_px=height_px,
        pixel_size_m=pixel_size_m,
        objective=obj_name,
        numerical_aperture=na,
        channels=channels,
        time_stamps_raw=time_stamps_raw
    )


def image_meta_to_dict(meta: ImageMeta) -> Dict[str, Any]:
    d = asdict(meta)
    # Turn ChannelMeta objects into dicts
    d["channels"] = {k: asdict(v) for k, v in meta.channels.items()}
    return d
