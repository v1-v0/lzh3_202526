import os

def select_directory(max_depth=2):
    """
    Lists directories up to 2 levels deep in the 'source' subfolder,
    but only if they have a direct sub-folder with a name that starts 
    with 'Control' or 'control'.
    Prompts the user to select one by entering its number or relative path,
    and returns the selected full relative path from the application root.
    """
    root_dir = 'source'
    directories = []
    
    # Walk through the directory up to max_depth + 1
    for root, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == '.':
            depth = 0
        else:
            depth = rel_path.count(os.sep) + 1
        
        if depth > max_depth + 1:
            dirs[:] = []
            continue
        
        if depth > 0:
            directories.append(rel_path.replace(os.sep, '\\'))
    
    # Find directories that have a direct 'Control' subfolder
    valid_directories = []
    for dir_path in directories:
        # Only check directories up to max_depth
        if len(dir_path.split('\\')) > max_depth:
            continue
            
        # Check if this directory has a direct child starting with 'Control'
        full_path = os.path.join(root_dir, dir_path.replace('\\', os.sep))
        if os.path.isdir(full_path):
            try:
                subdirs = [d for d in os.listdir(full_path) 
                          if os.path.isdir(os.path.join(full_path, d))]
                has_control = any(d.lower().startswith('control') for d in subdirs)
                if has_control:
                    valid_directories.append(dir_path)
            except OSError:
                continue
    
    # Remove duplicates and sort
    directories = sorted(set(valid_directories))
    
    # Display the list
    if not directories:
        print("No directories found.")
        return None
    
    print("Available directories (up to 2 levels deep):")
    for i, dir_path in enumerate(directories, 1):
        print(f"{i}) {dir_path}")
    
    # Prompt user to select
    while True:
        selected = input("\nEnter the number or full path of the directory you want to select: ").strip()
        
        # Check if input is a number
        if selected.isdigit():
            num = int(selected)
            if 1 <= num <= len(directories):
                selected_path = directories[num - 1]
                # Return full relative path from application root
                return os.path.join(root_dir, selected_path.replace('\\', os.sep))
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(directories)}.")
        # Check if input is a valid path
        elif selected in directories:
            # Return full relative path from application root
            return os.path.join(root_dir, selected.replace('\\', os.sep))
        else:
            print("Invalid selection. Please enter a valid number or path.")

def main() -> None:
    selected_directory = select_directory()
    if selected_directory:
        print(f"\nYou selected: {selected_directory}")
    else:
        print("No directory selected.")

if __name__ == "__main__":
    main()