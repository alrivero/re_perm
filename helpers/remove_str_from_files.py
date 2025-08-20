import os
import argparse
from pathlib import Path

def main():
    """Main function to parse arguments and rename files."""
    parser = argparse.ArgumentParser(
        description="Removes one or more substrings from filenames in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-d', '--dir', type=str, required=True,
        help='Path to the directory containing files to rename.'
    )
    parser.add_argument(
        '-s', '--substrings', type=str, nargs='+', required=True,
        help='One or more substrings to remove from the filenames. \n(e.g., -s "_copy" ".tmp" " (1)")'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what changes would be made without actually renaming files.'
    )
    args = parser.parse_args()

    target_dir = Path(args.dir)

    if not target_dir.is_dir():
        print(f"❌ Error: Directory not found at '{target_dir}'")
        return

    if args.dry_run:
        print("--- DRY RUN MODE ---")
        print("No files will actually be renamed.\n")

    files_renamed = 0
    files_to_rename = 0

    # Iterate through all items in the target directory
    for item_path in target_dir.iterdir():
        original_name = item_path.name
        new_name = original_name

        # Sequentially remove each specified substring
        for sub in args.substrings:
            new_name = new_name.replace(sub, '')

        # Proceed only if the name has changed
        if new_name != original_name:
            files_to_rename += 1
            new_path = item_path.with_name(new_name)

            # Safety Check: prevent overwriting an existing file
            if new_path.exists():
                print(f"⚠️ SKIPPED: Cannot rename '{original_name}' to '{new_name}' because the destination already exists.")
                continue

            if args.dry_run:
                print(f"Would rename '{original_name}' to '{new_name}'")
            else:
                try:
                    item_path.rename(new_path)
                    print(f"Renamed '{original_name}' to '{new_name}'")
                    files_renamed += 1
                except OSError as e:
                    print(f"❌ ERROR: Could not rename '{original_name}': {e}")

    # --- Final Summary ---
    print("\n--- Summary ---")
    if args.dry_run:
        print(f"Found {files_to_rename} files that would be renamed.")
    else:
        print(f"✅ Process complete. {files_renamed} out of {files_to_rename} targeted files were successfully renamed.")

if __name__ == '__main__':
    main()