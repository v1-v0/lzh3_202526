import os
import glob
import xml.etree.ElementTree as ET

class DataLoader:
    def __init__(self, root_path, sample_folder_name):
        self.root_path = root_path
        self.control_folder = os.path.join(root_path, "Control group")
        self.sample_folder = os.path.join(root_path, sample_folder_name)
        
        # File suffixes
        self.bf_suffix = "_ch00.tif"
        self.fl_suffix = "_ch01.tif"
        self.meta_folder_name = "MetaData"
        self.xml_suffix = ".xml" 

        # Validation Keywords extracted from "Metadata Microscopy Specifications.docx"
        # The XML must contain at least one of these to be considered the CORRECT metadata.
        self.required_metadata_keywords = [
            "Leica",            # Manufacturer
            "DMI8",             # Microscope Model
            "Prime 95B",        # Camera Model
            "N PLAN 100x",      # Objective
            "11506158"          # Objective Number
        ]

    def _is_valid_xml(self, xml_path):
        """
        Validates XML based on structure AND content from the Specifications DOCX.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 1. Basic Structure Check
            if root is None:
                return False, "XML root is empty"
            
            # 2. Content Verification against Specifications
            # We convert the XML tree to a string to search for keywords defined in your DOCX
            xml_string = ET.tostring(root, encoding='unicode')
            
            found_keyword = False
            for keyword in self.required_metadata_keywords:
                if keyword in xml_string:
                    found_keyword = True
                    break
            
            if not found_keyword:
                return False, "Valid XML, but missing Microscopy Specs (e.g., 'Leica', 'DMI8')"

            return True, "Valid"
            
        except ET.ParseError:
            return False, "XML syntax error (corrupted file)"
        except Exception as e:
            return False, f"Read error: {str(e)}"

    def debug_print_manifest(self, directory):
        """
        Short-term debug tool: Lists ALL files found in the directory structure
        so you can visually verify file naming conventions.
        """
        print(f"\n--- DEBUG MANIFEST FOR: {os.path.basename(directory)} ---")
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return

        # List all files recursively
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Create a relative path for cleaner display
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                all_files.append(rel_path)
        
        all_files.sort()
        
        if not all_files:
            print("  (Folder is empty)")
        else:
            for f in all_files:
                print(f"  Found: {f}")
        print("---------------------------------------------------\n")

    def get_file_pairs(self, directory):
        valid_sets = []
        errors = []

        if not os.path.exists(directory):
            return [], [f"Directory missing: {directory}"]

        # Get Bright-Field images
        bf_files = glob.glob(os.path.join(directory, f"*{self.bf_suffix}"))
        
        for bf_path in bf_files:
            base_name = os.path.basename(bf_path).replace(self.bf_suffix, "")
            dir_name = os.path.dirname(bf_path)
            
            fl_path = os.path.join(dir_name, f"{base_name}{self.fl_suffix}")
            xml_path = os.path.join(dir_name, self.meta_folder_name, f"{base_name}{self.xml_suffix}")

            missing_components = []
            
            # Check FL image
            if not os.path.exists(fl_path):
                missing_components.append("Missing FL Image")

            # Check XML existence and content
            if not os.path.exists(xml_path):
                missing_components.append("Missing XML File")
            else:
                is_valid, msg = self._is_valid_xml(xml_path)
                if not is_valid:
                    missing_components.append(f"Invalid XML: {msg}")

            if not missing_components:
                valid_sets.append({
                    "id": base_name,
                    "bf": bf_path,
                    "fl": fl_path,
                    "xml": xml_path
                })
            else:
                errors.append(f"{base_name}: {', '.join(missing_components)}")

        return valid_sets, errors

    def validate_run_configuration(self):
        print("==========================================")
        print("      MICROSCOPY DATA VALIDATION          ")
        print("==========================================")
        
        # 1. DEBUG: Print raw file lists first
        self.debug_print_manifest(self.control_folder)
        self.debug_print_manifest(self.sample_folder)

        # 2. Process Control Group
        print(f"Processing Control Group...")
        control_data, control_errs = self.get_file_pairs(self.control_folder)
        
        # 3. Process Sample Group
        print(f"Processing Sample Group...")
        sample_data, sample_errs = self.get_file_pairs(self.sample_folder)

        # 4. Final Report
        print("\n==========================================")
        print("           VALIDATION SUMMARY             ")
        print("==========================================")
        
        print(f"Control Group: {len(control_data)} valid sets.")
        print(f"Sample Group:  {len(sample_data)} valid sets.")

        if control_errs or sample_errs:
            print("\n--- ERRORS DETECTED ---")
            for e in control_errs: print(f"[Control] {e}")
            for e in sample_errs: print(f"[Sample]  {e}")
            return False
        
        print("\nSUCCESS: All files match the Microscopy Specifications.")
        return True

# ==========================================
# EXECUTION
# ==========================================

ROOT_DIRECTORY = os.path.join(os.getcwd(), "source")
TARGET_SAMPLE_FOLDER = "12" 

loader = DataLoader(ROOT_DIRECTORY, TARGET_SAMPLE_FOLDER)
loader.validate_run_configuration()
