import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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

def compare_microgel_groups(source_root: str, sample_name: str) -> Dict:
    """
    Compare Microgel Negative (G-) and Microgel Positive (G+) groups within a sample 
    to check if they have matching file counts.
    
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
    
    # Get counts by folder
    neg_folder_counts = count_channel_files_by_folder(microgel_negative_path)
    pos_folder_counts = count_channel_files_by_folder(microgel_positive_path)
    
    results['microgel_negative'] = {
        'ch00_count': neg_ch00,
        'ch01_count': neg_ch01,
        'total_files': neg_ch00 + neg_ch01,
        'subdirs': neg_subdirs,
        'folder_counts': neg_folder_counts
    }
    
    results['microgel_positive'] = {
        'ch00_count': pos_ch00,
        'ch01_count': pos_ch01,
        'total_files': pos_ch00 + pos_ch01,
        'subdirs': pos_subdirs,
        'folder_counts': pos_folder_counts
    }
    
    # Check if counts match
    results['ch00_match'] = neg_ch00 == pos_ch00
    results['ch01_match'] = neg_ch01 == pos_ch01
    results['total_match'] = (neg_ch00 + neg_ch01) == (pos_ch00 + pos_ch01)
    results['ch00_ch01_paired_negative'] = neg_ch00 == neg_ch01
    results['ch00_ch01_paired_positive'] = pos_ch00 == pos_ch01
    
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

def print_comparison_report(results: Dict):
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
    
    # Detailed folder breakdown
    if results['folder_comparison']:
        print(f"\n{'-'*80}")
        print("FOLDER-BY-FOLDER BREAKDOWN:")
        print(f"{'-'*80}")
        print(f"{'Folder':<40} | {'G- ch00':<8} | {'G- ch01':<8} | {'G+ ch00':<8} | {'G+ ch01':<8} | {'Match'}")
        print(f"{'-'*80}")
        
        for folder, comparison in results['folder_comparison'].items():
            neg_ch00 = comparison['negative']['ch00']
            neg_ch01 = comparison['negative']['ch01']
            pos_ch00 = comparison['positive']['ch00']
            pos_ch01 = comparison['positive']['ch01']
            match_status = "✓" if comparison['match'] else "✗"
            
            folder_display = folder[:38] + '..' if len(folder) > 40 else folder
            print(f"{folder_display:<40} | {neg_ch00:<8} | {neg_ch01:<8} | {pos_ch00:<8} | {pos_ch01:<8} | {match_status}")
    
    # Final verdict
    print(f"\n{'-'*80}")
    if (results['ch00_match'] and results['ch01_match'] and 
        results['ch00_ch01_paired_negative'] and results['ch00_ch01_paired_positive']):
        print("✓ PASS: G- and G+ have matching file counts and proper ch00/ch01 pairing!")
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
        
        print(f"✗ FAIL: Issues found - {', '.join(issues)}")
    print(f"{'='*80}\n")

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

def run_comparison(source_root: str = './source'):
    """Main function to check samples in the source directory based on user selection."""
    
    # Check if source directory exists
    if not Path(source_root).exists():
        print(f"Error: Source directory '{source_root}' not found!")
        return
    
    # Prompt user for directory selection
    selected_directory = prompt_directory_selection(source_root)
    
    if selected_directory is None:
        return
    
    all_results = {}
    
    if selected_directory == 'ALL':
        # Check all available directories
        directories = get_available_directories(source_root)
        print(f"\nChecking all {len(directories)} directories...\n")
        
        for directory in directories:
            results = compare_microgel_groups(source_root, directory)
            all_results[directory] = results
            print_comparison_report(results)
    else:
        # Check only the selected directory
        results = compare_microgel_groups(source_root, selected_directory)
        all_results[selected_directory] = results
        print_comparison_report(results)
    
    # Generate overall summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Sample':<30} | {'Count Match':<15} | {'Pairing'}")
        print(f"{'-'*80}")
        
        for sample, results in all_results.items():
            if 'error' not in results:
                count_match = "✓ MATCH" if (results['ch00_match'] and results['ch01_match']) else "✗ MISMATCH"
                pairing = "✓ PAIRED" if (results['ch00_ch01_paired_negative'] and 
                                         results['ch00_ch01_paired_positive']) else "✗ NOT PAIRED"
                print(f"{sample:<30} | {count_match:<15} | {pairing}")
            else:
                print(f"{sample:<30} | {'ERROR':<15} | ERROR")
        print(f"{'='*80}\n")

# Run the comparison
run_comparison('./source')