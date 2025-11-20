import os
import os
import pandas as pd


def fix_paths(annotation_csv: str, new_path: str):
    df = pd.read_csv(annotation_csv, delimiter=';')

    if 'Filename' not in df.columns:
        raise ValueError("CSV does not contain a 'Filename' column")

    def join_new_root(pat):
        if pd.isna(pat):
            return pat
        base = os.path.basename(pat)
        return os.path.join(new_path, base)

    df = df.assign(Filename=df['Filename'].apply(join_new_root))

    # overwrite the CSV using semicolon delimiter
    df.to_csv(annotation_csv, index=False, sep=';')
    print(f"Updated {len(df)} rows in {annotation_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LISA Traffic Light dataset annotations")
    parser.add_argument('annotation_csv', help='Path to annotations CSV file')
    parser.add_argument('new_path', help='New root path to prepend to image filenames')
    parser.add_argument('--dry-run', action='store_true', help='Print preview without writing changes')
    parser.add_argument('--backup', action='store_true', help='Create a .bak copy of the original CSV before writing')
    args = parser.parse_args()

    if args.dry_run:
        df = pd.read_csv(args.annotation_csv, delimiter=';')
        print('Original sample:')
        print(df.head())
        print('\nPreview of updated filenames:')
        print(df['Filename'].apply(lambda p: os.path.join(args.new_path, os.path.basename(p))).head())
    else:
        if args.backup:
            bak = args.annotation_csv + '.bak'
            with open(args.annotation_csv, 'rb') as src, open(bak, 'wb') as dst:
                dst.write(src.read())
            print(f'Backup created at {bak}')
        fix_paths(args.annotation_csv, args.new_path)
        print(f"Updated paths in {args.annotation_csv} to have new root {args.new_path}")
