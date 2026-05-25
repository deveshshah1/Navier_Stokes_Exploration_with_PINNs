import os
import numpy as np
import pandas as pd
from pathlib import Path

def read_openfoam_vector_field(filepath, n_cells):
    """Read an OpenFOAM vector field file (e.g. U) and return numpy array."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    start = content.find('internalField')
    line_start = content.find('\n', start) + 1
    line_end = content.find('\n', line_start)
    n_entries = int(content[line_start:line_end].strip())
    
    block_start = content.find('(', line_end) + 1
    block_end = content.rfind(')')
    block = content[block_start:block_end].strip()
    
    values = []
    for line in block.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            vals = list(map(float, line[1:-1].split()))
            if len(vals) == 3:
                values.append(vals)
        if len(values) == n_entries:
            break
    
    return np.array(values)

def read_openfoam_scalar_field(filepath, n_cells):
    """Read an OpenFOAM scalar field file (e.g. p) and return numpy array."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    start = content.find('internalField')
    line_start = content.find('\n', start) + 1
    line_end = content.find('\n', line_start)
    n_entries = int(content[line_start:line_end].strip())
    
    block_start = content.find('(', line_end) + 1
    block_end = content.rfind(')')
    block = content[block_start:block_end].strip()
    
    values = []
    for line in block.split('\n'):
        line = line.strip()
        if line and not line.startswith('('):
            try:
                values.append(float(line))
            except ValueError:
                continue
        if len(values) == n_entries:
            break
    
    return np.array(values)

def read_cell_centers(case_dir):
    """Read cell center coordinates from any time directory that has C file."""
    case_path = Path(case_dir)
    
    # Find any NON-ZERO time directory that has a C file
    c_file = None
    time_dirs = []
    for d in case_path.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append((t, d))
            except ValueError:
                continue
    
    for t, d in sorted(time_dirs):
        candidate = d / 'C'
        if candidate.exists():
            c_file = candidate
            break
    
    if c_file is None:
        raise FileNotFoundError(
            f"No C file found in any time directory under {case_dir}\n"
            f"Run this first in your OpenFOAM case:\n"
            f"  postProcess -func writeCellCentres"
        )
    
    print(f"  Reading cell centers from {c_file}")
    
    with open(c_file, 'r') as f:
        content = f.read()
    
    # Find internalField block
    start = content.find('internalField')
    line_start = content.find('\n', start) + 1
    line_end = content.find('\n', line_start)
    n_entries = int(content[line_start:line_end].strip())
    
    block_start = content.find('(', line_end) + 1
    block_end = content.rfind(')')
    block = content[block_start:block_end].strip()
    
    coords = []
    for line in block.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            vals = list(map(float, line[1:-1].split()))
            if len(vals) == 3:
                coords.append(vals)
        # Stop once we have enough internal cell centers
        if len(coords) == n_entries:
            break
    
    arr = np.array(coords)
    print(f"  Parsed {len(arr)} internal cell centers")
    return arr

def get_time_directories(case_dir):
    """Get all numeric time directories in sorted order."""
    case_path = Path(case_dir)
    time_dirs = []
    for d in case_path.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:  # skip t=0
                    time_dirs.append((t, d))
            except ValueError:
                continue
    return sorted(time_dirs, key=lambda x: x[0])

def extract_case(case_dir, Re, output_path):
    """Extract all field data from an OpenFOAM case and save as parquet."""
    print(f"\n{'='*50}")
    print(f"Extracting Re={Re} from {case_dir}")
    print(f"{'='*50}")
    
    case_path = Path(case_dir)
    
    # Read cell centers (x, y, z coordinates)
    print("Reading cell centers...")
    coords = read_cell_centers(case_dir)
    x = coords[:, 0]
    y = coords[:, 1]
    # z is ignored for 2D
    n_cells = len(x)
    print(f"  Found {n_cells} cells")
    
    # Get all time directories
    time_dirs = get_time_directories(case_dir)
    print(f"  Found {len(time_dirs)} time snapshots")
    print(f"  Time range: {time_dirs[0][0]:.3f} to {time_dirs[-1][0]:.3f}")
    
    # Build dataframe across all timesteps
    all_records = []
    
    for i, (t, tdir) in enumerate(time_dirs):
        if i % 10 == 0:
            print(f"  Processing t={t:.3f} ({i+1}/{len(time_dirs)})")
        
        u_file = tdir / 'U'
        p_file = tdir / 'p'
        
        if not u_file.exists() or not p_file.exists():
            print(f"  Warning: skipping t={t}, missing U or p")
            continue
        
        try:
            U = read_openfoam_vector_field(u_file, n_cells)
            p = read_openfoam_scalar_field(p_file, n_cells)
        except Exception as e:
            print(f"  Warning: skipping t={t}, error: {e}")
            continue
        
        if len(U) != n_cells or len(p) != n_cells:
            print(f"  Warning: skipping t={t}, field size mismatch")
            continue
        
        df_t = pd.DataFrame({
            'x':  x,
            'y':  y,
            't':  t,
            'Ux': U[:, 0],
            'Uy': U[:, 1],
            'p':  p,
            'Re': Re,
        })
        all_records.append(df_t)
    
    print(f"  Combining {len(all_records)} snapshots...")
    df = pd.concat(all_records, ignore_index=True)
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    return df

def main():
    # --- Configure paths here ---
    RE20_CASE  = '/Users/deveshshah/Documents/Programming/openfoam-data/cylinder_Re20'
    RE100_CASE = '/Users/deveshshah/Documents/Programming/openfoam-data/cylinder_Re100'
    OUTPUT_DIR = './dataset/'

    # Extract Re=20
    df20 = extract_case(
        case_dir=RE20_CASE,
        Re=20,
        output_path=f'{OUTPUT_DIR}/cylinder_Re20_groundtruth.parquet'
    )

    # Extract Re=100
    df100 = extract_case(
        case_dir=RE100_CASE,
        Re=100,
        output_path=f'{OUTPUT_DIR}/cylinder_Re100_groundtruth.parquet'
    )

    # Print final summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    for Re, df in [(20, df20), (100, df100)]:
        print(f"\nRe={Re}:")
        print(f"  Snapshots: {df['t'].nunique()}")
        print(f"  Cells per snapshot: {df[df['t']==df['t'].iloc[0]].shape[0]}")
        print(f"  Time range: {df['t'].min():.3f} to {df['t'].max():.3f}")
        print(f"  Ux range: {df['Ux'].min():.4f} to {df['Ux'].max():.4f}")
        print(f"  Uy range: {df['Uy'].min():.4f} to {df['Uy'].max():.4f}")
        print(f"  p range:  {df['p'].min():.4f} to {df['p'].max():.4f}")

if __name__ == '__main__':
    main()