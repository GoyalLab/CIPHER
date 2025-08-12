import os
import numpy as np
import pandas as pd
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from src.preprocess import get_data


def split_data(adata_p: str, data_dir: str, n_splits: int = 10, seed: int = 0, out_file_name: str = 'perturb_processed.h5ad', qc: bool = False, dry: bool = False) -> pd.DataFrame:
    # I/O
    ds_name = os.path.basename(adata_p).replace('.h5ad', '')        # Dataset name
    # Base output directory
    output_dir = os.path.join(data_dir, f'{ds_name}_splits')         # {data_dir}/{ds_name}_splits/
    # Set summary output file
    summary_f = os.path.join(output_dir, 'adata_summary.csv')
    # Set seed for reproducibilty
    np.random.seed(seed)
    adata, _, _ = get_data(0, adata_p, qc=qc, save=False)
    # Split dataset by perturbations and number of cells
    cpp = adata.obs.perturbation.value_counts()
    # Choose perturbations without control (is always included)
    cppnc = cpp[cpp.index!='control']
    # Randomly shuffle perturbations to get a even mix
    r_cppnc = cppnc[np.random.permutation(cppnc.index)]
    # Split into N arrays of equal size
    splits = np.array_split(r_cppnc.index, n_splits)
    # Calculate control mask
    ctrl_mask = adata.obs.perturbation=='control'
    # Create adata summary dataframe to update
    adata_summary = []
    # Create increasingly larger datasets from start split to end
    prev_mask = None
    for i, s in enumerate(splits):
        # Create output directory for split
        split_dir = os.path.join(output_dir, f'split_{i}')
        os.makedirs(split_dir, exist_ok=True)
        # Set output adata path
        o = os.path.join(split_dir, out_file_name)
        # Subset to perturbations in current split
        c_mask = (adata.obs.perturbation.isin(s))
        n_cells = c_mask.sum()
        # Add previous split
        if prev_mask is None:
            prev_mask = c_mask
        else:
            c_mask += prev_mask
            prev_mask = c_mask
        # Add control mask
        mask = c_mask | (ctrl_mask)
        # Subset adata to total mask
        tmp = adata[mask]
        logging.info(f'Creating split {i} with shape: {tmp.shape} and #perturbations: {tmp.obs.perturbation.nunique()}')
        # Add split summary to overall data summary
        adata_split_summary = {'split': f'split_{i}', 'n_obs': tmp.n_obs, 'n_perts': tmp.obs.perturbation.nunique(), 'n_new_cells': n_cells}
        adata_summary.append(adata_split_summary)
        if not dry:
            # Write split to file with compression to reduce disk usage
            tmp.copy().write_h5ad(o, compression='gzip')
    # Combine summary
    adata_summary = pd.DataFrame(adata_summary)
    if not dry:
        adata_summary.to_csv(summary_f)
    return adata_summary