# extract_all_metadata.py
# Run once → python extract_all_metadata.py
# Creates metadata_json/10 P 1.json … 19 P 5.json automatically

import xml.etree.ElementTree as ET
from pathlib import Path
import json

SOURCE_ROOT = Path("source")
OUTPUT_DIR = Path("metadata_json")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_sample_name(xml_path: Path):
    """Extract sample name from XML file path (e.g., '10 P 1' from '10 P 1.xml')."""
    return xml_path.stem

def extract_channel_scaling(root):
    """
    Extract scaling information for Gray and Red channels from XML root.
    
    Returns:
    tuple: (scaling_info, intensity_info) dictionaries
    """
    # Find the ViewerScaling attachment
    viewer_scaling = root.find(".//Attachment[@Name='ViewerScaling']")
    
    if viewer_scaling is None:
        return None, None
    
    # Get all ChannelScalingInfo elements
    channel_scaling_infos = viewer_scaling.findall('ChannelScalingInfo')
    
    # Extract scaling information
    scaling_info = {}
    intensity_info = {}
    max_value = 4095  # 12-bit
    
    # Channel names based on LUT
    channel_names = ['Gray', 'Red']
    
    for idx, channel_info in enumerate(channel_scaling_infos):
        if idx >= len(channel_names):
            break
            
        channel_name = channel_names[idx]
        
        black_val = float(channel_info.get('BlackValue', 0))
        white_val = float(channel_info.get('WhiteValue', 0))
        gamma_val = float(channel_info.get('GammaValue', 1))
        automatic = channel_info.get('Automatic', '0') == '1'
        lut_name = channel_info.get('BackgroundLutName', '')
        
        scaling_info[channel_name] = {
            'BackgroundLutName': lut_name,
            'BlackValue': black_val,
            'WhiteValue': white_val,
            'GammaValue': gamma_val,
            'Automatic': automatic
        }
        
        intensity_info[channel_name] = {
            'BackgroundLutName': lut_name,
            'BlackValue_normalized': black_val,
            'WhiteValue_normalized': white_val,
            'BlackValue_intensity': int(round(black_val * max_value)),
            'WhiteValue_intensity': int(round(white_val * max_value)),
            'GammaValue': gamma_val,
            'Automatic': automatic
        }
    
    return scaling_info, intensity_info

def extract_metadata(xml_path: Path):
    """Extract comprehensive metadata from LAS AF XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Sample name
        sample_name = extract_sample_name(xml_path)

        # Image name
        name_elem = root.find(".//Name")
        image_name = name_elem.text.strip() if name_elem is not None and name_elem.text else sample_name

        # Pixel size in µm - Fix deprecation warning
        dim = root.find(".//DimensionDescription[@DimID='1']")
        if dim is None:
            dim = root.find(".//DimensionDescription[@DimID='X']")
        
        if dim is not None:
            pixels = int(dim.get("NumberOfElements", "1024"))
            length_m = float(dim.get("Length", "0.000132"))
            pixel_size_um = length_m * 1e6 / pixels
        else:
            pixel_size_um = 0.1289  # Default fallback

        # Channel scaling information
        scaling_info, intensity_info = extract_channel_scaling(root)
        
        # Fallback values if extraction fails
        gray_b_12 = gray_w_12 = red_b_12 = red_w_12 = 0
        
        if intensity_info:
            if 'Gray' in intensity_info:
                gray_b_12 = intensity_info['Gray']['BlackValue_intensity']
                gray_w_12 = intensity_info['Gray']['WhiteValue_intensity']
            if 'Red' in intensity_info:
                red_b_12 = intensity_info['Red']['BlackValue_intensity']
                red_w_12 = intensity_info['Red']['WhiteValue_intensity']

        # Acquisition time
        time_elem = root.find(".//StartTime")
        acq_time = time_elem.text if time_elem is not None and time_elem.text else ""

        # Exposure times
        bf_exp = fluo_exp = 138.0
        for ch in root.findall(".//WideFieldChannelInfo"):
            exp = ch.get("ExposureTime", "")
            if "ms" in exp:
                val = float(exp.replace(" ms", ""))
                if ch.get("ContrastingMethodName") == "TL-BF":
                    bf_exp = val
                elif "FLUO" in ch.get("Contrast", ""):
                    fluo_exp = val

        # Construct complete metadata
        metadata = {
            "sample_name": sample_name,
            "image_name": image_name,
            "folder": str(xml_path.parent.parent.name),  # "10", "11", ..., "19"
            "acquired": acq_time,
            "pixel_size_um": round(pixel_size_um, 4),
            "objective": "N PLAN 100x/1.25 OIL",
            "exposure_times": {
                "brightfield_ms": bf_exp,
                "fluorescence_ms": fluo_exp
            },
            "channels": {}
        }

        # Add channel-specific information
        if scaling_info and intensity_info:
            for channel in scaling_info.keys():
                metadata["channels"][channel] = {
                    "BackgroundLutName": scaling_info[channel]['BackgroundLutName'],
                    "normalized": {
                        "BlackValue": scaling_info[channel]['BlackValue'],
                        "WhiteValue": scaling_info[channel]['WhiteValue']
                    },
                    "intensity_12bit": {
                        "BlackValue": intensity_info[channel]['BlackValue_intensity'],
                        "WhiteValue": intensity_info[channel]['WhiteValue_intensity']
                    },
                    "GammaValue": scaling_info[channel]['GammaValue'],
                    "Automatic": scaling_info[channel]['Automatic']
                }
        
        # Add recommended thresholds (backward compatible)
        metadata["recommended_thresholds"] = {
            "brightfield": (gray_b_12 + gray_w_12) // 2 if gray_w_12 > 0 else 500,
            "fluorescence": red_w_12 if red_w_12 > 0 else 300
        }
        
        # Legacy fields for backward compatibility
        metadata["display_gray_black_12bit"] = gray_b_12
        metadata["display_gray_white_12bit"] = gray_w_12
        metadata["display_red_white_12bit"] = red_w_12
        metadata["recommended_threshold_bf"] = metadata["recommended_thresholds"]["brightfield"]
        metadata["recommended_threshold_fluo"] = metadata["recommended_thresholds"]["fluorescence"]

        # Save to JSON
        out_file = OUTPUT_DIR / f"{sample_name}.json"
        out_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"✓ Extracted → {out_file.name}")

    except Exception as e:
        print(f"✗ Failed {xml_path.name}: {e}")

if __name__ == "__main__":
    print("="*60)
    print("EXTRACTING METADATA FOR FOLDERS 10 → 19")
    print("="*60 + "\n")
    
    total_files = 0
    successful = 0
    
    for folder in range(10, 20):  # 10 to 19 inclusive
        folder_path = SOURCE_ROOT / str(folder)
        metadata_folder = folder_path / "MetaData"
        
        if not metadata_folder.exists():
            print(f"⊘ Skipping folder {folder} (no MetaData folder)")
            continue

        xml_files = sorted(metadata_folder.glob("*.xml"))
        
        # Filter out hidden files (starting with ._) and Properties files
        xml_files = [
            f for f in xml_files 
            if not f.name.startswith("._") and "Properties" not in f.name
        ]
        
        if not xml_files:
            print(f"⊘ No valid XML files in folder {folder}")
            continue

        print(f"\nProcessing folder {folder} ({len(xml_files)} images)")
        print("-" * 60)
        
        for xml_file in xml_files:
            total_files += 1
            extract_metadata(xml_file)
            successful += 1
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total files processed: {successful}/{total_files}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")