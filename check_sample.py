import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import re

def extract_file_identifier(filename: str) -> Optional[str]:
    """
    Extract the base identifier from a filename (everything before _ch00 or _ch01).
    
    Args:
        filename: Name of the file (e.g., '1 P NO 1_ch00.tif')
    
    Returns:
        Base identifier (e.g., '1 P NO 1')
    """
    # Remove the channel and extension
    match = re.match(r'(.+?)_ch0[01]\.tif$', filename)
    if match:
        return match.group(1)
    return None

def get_file_details(directory: Path) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Get detailed file information organized by subfolder and identifier.
    
    Args:
        directory: Directory to scan
    
    Returns:
        Nested dict: {subfolder: {identifier: [(channel, full_path), ...]}}
    """
    file_details = defaultdict(lambda: defaultdict(list))
    
    if not directory.exists():
        return {}
    
    for root, dirs, files in os.walk(directory):
        # Skip MetaData directories
        if 'MetaData' in root:
            continue
        
        relative_path = Path(root).relative_to(directory)
        folder_key = str(relative_path) if str(relative_path) != '.' else '(root)'
        
        for file in files:
            if file.endswith('_ch00.tif') or file.endswith('_ch01.tif'):
                identifier = extract_file_identifier(file)
                if identifier:
                    channel = 'ch00' if file.endswith('_ch00.tif') else 'ch01'
                    full_path = os.path.join(root, file)
                    file_details[folder_key][identifier].append((channel, full_path))
    
    return dict(file_details)

def count_channel_files(directory: Path) -> Tuple[int, int, List[str]]:
    """
    Count the number of ch00 and ch01 files in a directory and its subdirectories.
    
    Args:
        directory: Directory to scan
    
    Returns:
        Tuple of (ch00_count, ch01_count, list of all subdirectories scanned)
    """
    ch00_count = 0
    ch01_count = 0
    subdirs = []
    
    if not directory.exists():
        return 0, 0, []
    
    for root, dirs, files in os.walk(directory):
        # Skip MetaData directories
        if 'MetaData' in root:
            continue
        
        relative_path = Path(root).relative_to(directory)
        if str(relative_path) != '.':
            subdirs.append(str(relative_path))
        
        for file in files:
            if file.endswith('_ch00.tif'):
                ch00_count += 1
            elif file.endswith('_ch01.tif'):
                ch01_count += 1
    
    return ch00_count, ch01_count, sorted(set(subdirs))

def count_channel_files_by_folder(directory: Path) -> Dict[str, Tuple[int, int]]:
    """
    Count ch00 and ch01 files in each subfolder.
    
    Args:
        directory: Directory to scan
    
    Returns:
        Dictionary mapping folder_path -> (ch00_count, ch01_count)
    """
    folder_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    
    if not directory.exists():
        return {}
    
    for root, dirs, files in os.walk(directory):
        # Skip MetaData directories
        if 'MetaData' in root:
            continue
        
        relative_path = Path(root).relative_to(directory)
        folder_key = str(relative_path) if str(relative_path) != '.' else '(root)'
        
        for file in files:
            if file.endswith('_ch00.tif'):
                folder_counts[folder_key][0] += 1
            elif file.endswith('_ch01.tif'):
                folder_counts[folder_key][1] += 1
    
    # Convert to regular dict with tuples
    result: Dict[str, Tuple[int, int]] = {}
    for k, v in folder_counts.items():
        result[k] = (v[0], v[1])
    
    return result

def verify_identifier_pairing(neg_details: Dict, pos_details: Dict) -> Dict:
    """
    Verify that identifiers match between G- and G+ groups.
    
    Args:
        neg_details: File details from G- (Microgel Negative)
        pos_details: File details from G+ (Microgel Positive)
    
    Returns:
        Dictionary with pairing verification results
    """
    pairing_results = {
        'matched_folders': [],
        'missing_in_positive': [],
        'missing_in_negative': [],
        'folder_details': {}
    }
    
    all_folders = set(neg_details.keys()) | set(pos_details.keys())
    
    for folder in sorted(all_folders):
        neg_identifiers = set(neg_details.get(folder, {}).keys())
        pos_identifiers = set(pos_details.get(folder, {}).keys())
        
        matched = neg_identifiers & pos_identifiers
        missing_in_pos = neg_identifiers - pos_identifiers
        missing_in_neg = pos_identifiers - neg_identifiers
        
        folder_result = {
            'matched_count': len(matched),
            'matched_identifiers': sorted(matched),
            'missing_in_positive': sorted(missing_in_pos),
            'missing_in_negative': sorted(missing_in_neg),
            'perfect_match': len(missing_in_pos) == 0 and len(missing_in_neg) == 0
        }
        
        # Check channel completeness for matched identifiers
        incomplete_pairs = []
        for identifier in matched:
            neg_channels = set(ch for ch, _ in neg_details[folder][identifier])
            pos_channels = set(ch for ch, _ in pos_details[folder][identifier])
            
            if neg_channels != {'ch00', 'ch01'} or pos_channels != {'ch00', 'ch01'}:
                incomplete_pairs.append({
                    'identifier': identifier,
                    'neg_channels': sorted(neg_channels),
                    'pos_channels': sorted(pos_channels)
                })
        
        folder_result['incomplete_pairs'] = incomplete_pairs
        pairing_results['folder_details'][folder] = folder_result
        
        if folder_result['perfect_match'] and len(incomplete_pairs) == 0:
            pairing_results['matched_folders'].append(folder)
        if missing_in_pos:
            pairing_results['missing_in_positive'].append((folder, list(missing_in_pos)))
        if missing_in_neg:
            pairing_results['missing_in_negative'].append((folder, list(missing_in_neg)))
    
    return pairing_results

def compare_microgel_groups(source_root: str, sample_name: str) -> Dict:
    """
    Compare Microgel Negative (G-) and Microgel Positive (G+) groups within a sample 
    to check if they have matching file counts and identifiers.
    
    Args:
        source_root: Root directory containing samples
        sample_name: Name of the sample (e.g., 'PD sample' or 'Spike sample')
    
    Returns:
        Dictionary containing comparison results
    """
    source_path = Path(source_root)
    sample_path = source_path / sample_name
    
    microgel_negative_path = sample_path / 'G-'
    microgel_positive_path = sample_path / 'G+'
    
    results = {
        'sample_name': sample_name,
        'microgel_negative_exists': microgel_negative_path.exists(),
        'microgel_positive_exists': microgel_positive_path.exists(),
    }
    
    if not results['microgel_negative_exists'] or not results['microgel_positive_exists']:
        results['error'] = (f"Missing directories: "
                          f"G- (Microgel Negative) exists={results['microgel_negative_exists']}, "
                          f"G+ (Microgel Positive) exists={results['microgel_positive_exists']}")
        return results
    
    # Get overall counts
    neg_ch00, neg_ch01, neg_subdirs = count_channel_files(microgel_negative_path)
    pos_ch00, pos_ch01, pos_subdirs = count_channel_files(microgel_positive_path)
    
    # Get counts by folder - BUG FIX: was using microgel_negative_path for both
    neg_folder_counts = count_channel_files_by_folder(microgel_negative_path)
    pos_folder_counts = count_channel_files_by_folder(microgel_positive_path)  # Fixed!
    
    # Get detailed file information
    neg_file_details = get_file_details(microgel_negative_path)
    pos_file_details = get_file_details(microgel_positive_path)
    
    # Verify identifier pairing
    pairing_verification = verify_identifier_pairing(neg_file_details, pos_file_details)
    
    results['microgel_negative'] = {
        'ch00_count': neg_ch00,
        'ch01_count': neg_ch01,
        'total_files': neg_ch00 + neg_ch01,
        'subdirs': neg_subdirs,
        'folder_counts': neg_folder_counts,
        'file_details': neg_file_details
    }
    
    results['microgel_positive'] = {
        'ch00_count': pos_ch00,
        'ch01_count': pos_ch01,
        'total_files': pos_ch00 + pos_ch01,
        'subdirs': pos_subdirs,
        'folder_counts': pos_folder_counts,
        'file_details': pos_file_details
    }
    
    # Check if counts match
    results['ch00_match'] = neg_ch00 == pos_ch00
    results['ch01_match'] = neg_ch01 == pos_ch01
    results['total_match'] = (neg_ch00 + neg_ch01) == (pos_ch00 + pos_ch01)
    results['ch00_ch01_paired_negative'] = neg_ch00 == neg_ch01
    results['ch00_ch01_paired_positive'] = pos_ch00 == pos_ch01
    
    # Add pairing verification results
    results['pairing_verification'] = pairing_verification
    
    # Compare folder structure
    all_folders = set(neg_folder_counts.keys()) | set(pos_folder_counts.keys())
    results['folder_comparison'] = {}
    
    for folder in sorted(all_folders):
        neg_counts = neg_folder_counts.get(folder, (0, 0))
        pos_counts = pos_folder_counts.get(folder, (0, 0))
        results['folder_comparison'][folder] = {
            'negative': {'ch00': neg_counts[0], 'ch01': neg_counts[1]},
            'positive': {'ch00': pos_counts[0], 'ch01': pos_counts[1]},
            'match': neg_counts == pos_counts
        }
    
    return results

def print_comparison_report(results: Dict, show_details: bool = True):
    """Print a formatted report of the comparison results."""
    print(f"\n{'='*80}")
    print(f"Comparison Report for: {results['sample_name']}")
    print(f"{'='*80}\n")
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    # Overall summary
    print("OVERALL FILE COUNTS:")
    print(f"{'-'*80}")
    
    neg = results['microgel_negative']
    pos = results['microgel_positive']
    
    print(f"{'Directory':<30} | {'ch00 files':<12} | {'ch01 files':<12} | {'Total':<10}")
    print(f"{'-'*80}")
    print(f"{'G- (Microgel Negative)':<30} | {neg['ch00_count']:<12} | {neg['ch01_count']:<12} | {neg['total_files']:<10}")
    print(f"{'G+ (Microgel Positive)':<30} | {pos['ch00_count']:<12} | {pos['ch01_count']:<12} | {pos['total_files']:<10}")
    print(f"{'-'*80}")
    
    # Check results
    ch00_status = "✓ MATCH" if results['ch00_match'] else "✗ MISMATCH"
    ch01_status = "✓ MATCH" if results['ch01_match'] else "✗ MISMATCH"
    total_status = "✓ MATCH" if results['total_match'] else "✗ MISMATCH"
    
    print(f"{'ch00 files match:':<30} {ch00_status}")
    print(f"{'ch01 files match:':<30} {ch01_status}")
    print(f"{'Total files match:':<30} {total_status}")
    
    # Check pairing within each group
    print(f"\n{'-'*80}")
    print("CHANNEL PAIRING CHECK (ch00 vs ch01):")
    print(f"{'-'*80}")
    
    neg_pair_status = "✓ PAIRED" if results['ch00_ch01_paired_negative'] else "✗ NOT PAIRED"
    pos_pair_status = "✓ PAIRED" if results['ch00_ch01_paired_positive'] else "✗ NOT PAIRED"
    
    print(f"G- (Microgel Negative): {neg['ch00_count']} ch00 vs {neg['ch01_count']} ch01 - {neg_pair_status}")
    print(f"G+ (Microgel Positive): {pos['ch00_count']} ch00 vs {pos['ch01_count']} ch01 - {pos_pair_status}")
    
    # Identifier pairing verification
    pairing = results['pairing_verification']
    print(f"\n{'-'*80}")
    print("IDENTIFIER PAIRING VERIFICATION (G- vs G+):")
    print(f"{'-'*80}")
    
    all_matched = len(pairing['missing_in_positive']) == 0 and len(pairing['missing_in_negative']) == 0
    
    if all_matched:
        print("✓ All identifiers match between G- and G+ groups!")
        print(f"  Total folders with perfect matches: {len(pairing['matched_folders'])}")
    else:
        print("✗ Some identifiers are missing or mismatched!")
        
        if pairing['missing_in_positive']:
            print("\n  Missing in G+ (present in G- but not in G+):")
            for folder, identifiers in pairing['missing_in_positive']:
                print(f"    Folder '{folder}': {len(identifiers)} identifier(s)")
                if show_details:
                    for identifier in identifiers[:5]:  # Show first 5
                        print(f"      - {identifier}")
                    if len(identifiers) > 5:
                        print(f"      ... and {len(identifiers) - 5} more")
        
        if pairing['missing_in_negative']:
            print("\n  Missing in G- (present in G+ but not in G-):")
            for folder, identifiers in pairing['missing_in_negative']:
                print(f"    Folder '{folder}': {len(identifiers)} identifier(s)")
                if show_details:
                    for identifier in identifiers[:5]:  # Show first 5
                        print(f"      - {identifier}")
                    if len(identifiers) > 5:
                        print(f"      ... and {len(identifiers) - 5} more")
    
    # Check for incomplete channel pairs
    has_incomplete = False
    for folder, details in pairing['folder_details'].items():
        if details['incomplete_pairs']:
            if not has_incomplete:
                print(f"\n{'-'*80}")
                print("⚠ WARNING: Incomplete channel pairs detected:")
                print(f"{'-'*80}")
                has_incomplete = True
            
            print(f"\n  Folder '{folder}':")
            for pair in details['incomplete_pairs'][:5]:  # Show first 5
                print(f"    - {pair['identifier']}")
                print(f"      G- has: {', '.join(pair['neg_channels'])}")
                print(f"      G+ has: {', '.join(pair['pos_channels'])}")
            if len(details['incomplete_pairs']) > 5:
                print(f"    ... and {len(details['incomplete_pairs']) - 5} more")
    
    # Detailed folder breakdown
    if results['folder_comparison'] and show_details:
        print(f"\n{'-'*80}")
        print("FOLDER-BY-FOLDER BREAKDOWN:")
        print(f"{'-'*80}")
        print(f"{'Folder':<40} | {'G- ch00':<8} | {'G- ch01':<8} | {'G+ ch00':<8} | {'G+ ch01':<8} | {'IDs':<6} | {'Match'}")
        print(f"{'-'*80}")
        
        for folder in sorted(results['folder_comparison'].keys()):
            comparison = results['folder_comparison'][folder]
            neg_ch00 = comparison['negative']['ch00']
            neg_ch01 = comparison['negative']['ch01']
            pos_ch00 = comparison['positive']['ch00']
            pos_ch01 = comparison['positive']['ch01']
            
            # Get identifier match status
            folder_pairing = pairing['folder_details'].get(folder, {})
            id_status = "✓" if folder_pairing.get('perfect_match', False) else "✗"
            match_status = "✓" if (comparison['match'] and folder_pairing.get('perfect_match', False)) else "✗"
            
            folder_display = folder[:38] + '..' if len(folder) > 40 else folder
            print(f"{folder_display:<40} | {neg_ch00:<8} | {neg_ch01:<8} | {pos_ch00:<8} | {pos_ch01:<8} | {id_status:<6} | {match_status}")
    
    # Final verdict
    print(f"\n{'-'*80}")
    all_checks_pass = (
        results['ch00_match'] and 
        results['ch01_match'] and 
        results['ch00_ch01_paired_negative'] and 
        results['ch00_ch01_paired_positive'] and
        all_matched and
        not has_incomplete
    )
    
    if all_checks_pass:
        print("✓ PASS: Perfect match between G- and G+ groups!")
        print("  - File counts match")
        print("  - Channel pairing is correct")
        print("  - All identifiers match")
        print("  - No incomplete channel pairs")
    else:
        issues = []
        if not results['ch00_match']:
            issues.append("ch00 counts don't match")
        if not results['ch01_match']:
            issues.append("ch01 counts don't match")
        if not results['ch00_ch01_paired_negative']:
            issues.append("G- channels not paired")
        if not results['ch00_ch01_paired_positive']:
            issues.append("G+ channels not paired")
        if not all_matched:
            issues.append("identifier mismatch between G- and G+")
        if has_incomplete:
            issues.append("incomplete channel pairs detected")
        
        print(f"✗ FAIL: Issues found - {', '.join(issues)}")
    print(f"{'='*80}\n")

def export_pairing_details(results: Dict, output_file: Optional[str] = None):
    """
    Export detailed pairing information to a text file for verification.
    
    Args:
        results: Results dictionary from compare_microgel_groups
        output_file: Optional output file path. If None, generates based on sample name
    """
    if 'error' in results:
        print(f"Cannot export pairing details: {results['error']}")
        return
    
    if output_file is None:
        safe_name = results['sample_name'].replace(' ', '_').replace('/', '_')
        output_file = f"pairing_details_{safe_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Pairing Details for: {results['sample_name']}\n")
        f.write(f"{'='*100}\n\n")
        
        pairing = results['pairing_verification']
        neg_details = results['microgel_negative']['file_details']
        pos_details = results['microgel_positive']['file_details']
        
        for folder in sorted(pairing['folder_details'].keys()):
            folder_info = pairing['folder_details'][folder]
            f.write(f"\nFolder: {folder}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"Matched identifiers: {folder_info['matched_count']}\n")
            f.write(f"Perfect match: {folder_info['perfect_match']}\n\n")
            
            # Write matched identifiers with file paths
            if folder_info['matched_identifiers']:
                f.write("Matched Identifier Pairs:\n")
                for identifier in sorted(folder_info['matched_identifiers']):
                    f.write(f"\n  Identifier: {identifier}\n")
                    
                    # G- files
                    f.write(f"    G- files:\n")
                    for channel, filepath in sorted(neg_details[folder][identifier]):
                        f.write(f"      [{channel}] {filepath}\n")
                    
                    # G+ files
                    f.write(f"    G+ files:\n")
                    for channel, filepath in sorted(pos_details[folder][identifier]):
                        f.write(f"      [{channel}] {filepath}\n")
            
            # Write mismatches
            if folder_info['missing_in_positive']:
                f.write(f"\n  Missing in G+ (present in G-):\n")
                for identifier in sorted(folder_info['missing_in_positive']):
                    f.write(f"    - {identifier}\n")
                    for channel, filepath in sorted(neg_details[folder][identifier]):
                        f.write(f"        [{channel}] {filepath}\n")
            
            if folder_info['missing_in_negative']:
                f.write(f"\n  Missing in G- (present in G+):\n")
                for identifier in sorted(folder_info['missing_in_negative']):
                    f.write(f"    - {identifier}\n")
                    for channel, filepath in sorted(pos_details[folder][identifier]):
                        f.write(f"        [{channel}] {filepath}\n")
            
            f.write(f"\n{'='*100}\n")
    
    print(f"Pairing details exported to: {output_file}")

def get_available_directories(source_root: str) -> List[str]:
    """
    Get available directories in the source folder.
    
    Args:
        source_root: Root directory containing samples
    
    Returns:
        List of available directory names
    """
    source_path = Path(source_root)
    
    if not source_path.exists():
        return []
    
    directories = []
    
    for item in source_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            directories.append(item.name)
    
    return sorted(directories)

def prompt_directory_selection(source_root: str) -> Optional[str]:
    """
    Prompt user to select a directory from available options.
    
    Args:
        source_root: Root directory containing samples
    
    Returns:
        Selected directory name, 'ALL' to check all directories, or None if cancelled
    """
    directories = get_available_directories(source_root)
    
    if not directories:
        print(f"Error: No directories found in '{source_root}'")
        return None
    
    print(f"\n{'='*80}")
    print("Available directories (up to 1 level deep):")
    print(f"{'='*80}")
    
    for idx, dir_name in enumerate(directories, 1):
        print(f"  [{idx}] {dir_name}")
    
    print(f"  [0] Check all directories")
    print(f"  [q] Quit")
    print(f"{'='*80}\n")
    
    while True:
        try:
            choice = input("Select a directory to check (enter number or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Exiting...")
                return None
            
            choice_num = int(choice)
            
            if choice_num == 0:
                return 'ALL'
            elif 1 <= choice_num <= len(directories):
                selected = directories[choice_num - 1]
                print(f"\nYou selected: {selected}")
                return selected
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(directories)}, or 'q' to quit.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

def run_comparison(source_root: str = './source', export_details: bool = True):
    """Main function to check samples in the source directory based on user selection."""
    
    # Check if source directory exists
    if not Path(source_root).exists():
        print(f"Error: Source directory '{source_root}' not found!")
        return
    
    # Prompt user for directory selection
    selected_directory = prompt_directory_selection(source_root)
    
    if selected_directory is None:
        return
    
    all_results: Dict[str, Dict] = {}
    
    if selected_directory == 'ALL':
        # Check all available directories
        directories = get_available_directories(source_root)
        print(f"\nChecking all {len(directories)} directories...\n")
        
        for directory in directories:
            results = compare_microgel_groups(source_root, directory)
            all_results[directory] = results
            print_comparison_report(results, show_details=False)
            
            if export_details and 'error' not in results:
                export_pairing_details(results)
    else:
        # Check only the selected directory (it's a str, not None, due to early return above)
        results = compare_microgel_groups(source_root, selected_directory)
        all_results[selected_directory] = results
        print_comparison_report(results, show_details=True)
        
        if export_details and 'error' not in results:
            export_pairing_details(results)

    # Generate overall summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Sample':<30} | {'Count Match':<15} | {'Pairing':<15} | {'IDs Match'}")
        print(f"{'-'*80}")
        
        for sample, results in all_results.items():
            if 'error' not in results:
                count_match = "✓ MATCH" if (results['ch00_match'] and results['ch01_match']) else "✗ MISMATCH"
                pairing = "✓ PAIRED" if (results['ch00_ch01_paired_negative'] and 
                                         results['ch00_ch01_paired_positive']) else "✗ NOT PAIRED"
                
                pairing_ver = results['pairing_verification']
                ids_match = "✓ MATCH" if (len(pairing_ver['missing_in_positive']) == 0 and 
                                          len(pairing_ver['missing_in_negative']) == 0) else "✗ MISMATCH"
                
                print(f"{sample:<30} | {count_match:<15} | {pairing:<15} | {ids_match}")
            else:
                print(f"{sample:<30} | {'ERROR':<15} | {'ERROR':<15} | ERROR")
        print(f"{'='*80}\n")


# Run the comparison
if __name__ == "__main__":
    run_comparison('./source', export_details=True)