import os
import numpy as np
import pandas as pd

from typing import Literal
import logging


def timestamps_to_minutes_series(ts_series: pd.Series) -> pd.Series:
    """Convert a Series of timestamp strings to minutes since the first timestamp."""
    fmt = "%Y-%m-%d-%H:%M:%S"
    times = pd.to_datetime(ts_series, format=fmt)
    delta_minutes = (times - times.iloc[0]).dt.total_seconds() / 60
    return delta_minutes

class SstatTraceReader:
    UNITS = np.array(['KB', 'MB', 'GB'])
    # Define sstat columns
    Timestamp: str = 'Timestamp'
    DeltaMinutes: str = 'DeltaMinutes'
    MaxRss: str = 'MaxRSS'
    MaxRssFloat: str = 'MaxRssFloat'
    AveRSS: str = 'AveRSS'
    AveRSSFloat: str = 'AveRSSFloat'
    AveCPU: str = 'AveCPU'
    cols: list[str] = [Timestamp, MaxRss, AveRSS, AveCPU]
    NCols: int = 4


    def __init__(self, trace_path: str, target_unit: Literal['KB', 'MB', 'GB'] = 'GB', sep: str = ','):
        self.data = self._read_trace(trace_path, sep=sep)
        self.target_unit = target_unit
        # Setup trace data
        self._setup()

    def _is_valid_line(self, line: list[str]):
        # Invalid amount of columns in line, skip
        if len(line) != self.NCols:
            return False
        # Check if number is in correct format TODO: reformat it to K if its M or G            
        if np.any([(not l.endswith('K')) or l.endswith('+') for l in line[1:3]]):
            return False
        return True

    def _read_trace(self, trace_path: str, sep: str = ',', use_header: bool = True) -> pd.DataFrame:
        header = None
        data = []
        with open(trace_path, 'r') as tp:
            for i, line in enumerate(tp):
                # Read split line
                s = line.rstrip().split(sep)
                if i == 0 and use_header:
                    header = s
                    continue
                # Filter for valid lines in trace and include header
                if self._is_valid_line(s):
                    data.append(s)
        return pd.DataFrame(data, columns=header)

    def _setup(self) -> None:
        # Convert memory columns to float in target unit
        self.data[self.MaxRssFloat] = self.convert_kb(self.data[self.MaxRss])
        self.data[self.AveRSSFloat] = self.convert_kb(self.data[self.AveRSS])
        # Convert timestamps to minutes from starting point
        self.data[self.DeltaMinutes] = timestamps_to_minutes_series(self.data[self.Timestamp])
    
    def convert_kb(self, memory_str: pd.Series) -> None:
        """Convert sstat memory string to values in target unit"""
        unit_idx = np.where(self.target_unit == self.UNITS)[0]
        if len(unit_idx) == 0:
            raise ValueError(f'Target unit has to be one of: {self.UNITS}, got: {self.target_unit}')
        memory_kb = memory_str.str.rstrip('K').astype(int)
        memory_conv = memory_kb / (1024 ** unit_idx[0])
        return memory_conv
    
    def plot_mem_over_time(self, plt_path: str | None = None):
        """Plot memory usage over time."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(dpi=120)
        ax = sns.lineplot(self.data, x=self.DeltaMinutes, y=self.MaxRssFloat)
        plt.xlabel(f'Elapsed time in minutes')
        plt.ylabel(f'Memory usage in {self.target_unit}')
        if plt_path is None:
            plt.show()
        else:
            plt.savefig(plt_path, bbox_inches='tight', dpi=120)


class TraceData:
    reader_cls = SstatTraceReader

    def __init__(
        self,
        data_dir: str,
        methods: list[str] = ['gears', 'cipher'],
    ):
        self.data_dir = data_dir
        self.methods = methods
        # Read data
        self.data = self._get_data()
        # Aggregate data for summary
        self.agg_data = self._get_agg_data()
    
    def _get_data(self) -> pd.DataFrame:
        # Look for split datasets
        datasets = [d for d in os.listdir(self.data_dir) if d.endswith('splits')]
        # Load trace for every split in every dataset
        data = []
        for ds in datasets:
            logging.info(f'Collecting splits from {ds}')
            ds_dir = os.path.join(self.data_dir, ds)
            summary_f = os.path.join(ds_dir, 'adata_summary.csv')
            split_dirs = [d for d in os.listdir(ds_dir) if d.startswith('split_')]
            # Load each trace
            split_data = []
            # Sort split dirs by index
            split_dirs = pd.DataFrame(split_dirs, columns=['split_name'])
            split_dirs['pos'] = split_dirs['split_name'].str.split('_').str[1].astype(str)
            split_dirs.sort_values('pos', inplace=True)
            # Add memory trace for every split
            for i, split_dir in enumerate(split_dirs.split_name):
                # Define file paths in split dir
                full_split_dir = os.path.join(ds_dir, split_dir)
                # Read trace for each method
                for method in self.methods:
                    sstat_trace_f = os.path.join(full_split_dir, 'traces', f'{method}.csv')
                    # Read memory trace of split
                    if os.path.exists(sstat_trace_f):
                        logging.info(f'\t - Collecting trace {i} for method {method}')
                        trace_data = self.reader_cls(sstat_trace_f).data
                        trace_data['split_idx'] = i
                        trace_data['split'] = f'split_{i}'
                        trace_data['method'] = method
                        split_data.append(trace_data)
            if len(split_data) > 0:
                split_data = pd.concat(split_data, axis=0)
                split_data['dataset'] = ds
                # Read split summary and add to data
                summary = pd.read_csv(summary_f, index_col=0)
                split_data = split_data.merge(summary, on='split', how='left')
                data.append(split_data)
            else:
                logging.info(f'No split traces found for {ds}')
        if len(data) > 0:
            data = pd.concat(data, axis=0)
            return data
        else:
            return pd.DataFrame()
        
    def _get_agg_data(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError('TraceData object is not initialized.')
        # Get max values from each run
        agg_data = self.data.groupby(['method', 'split']).apply(
            lambda x: pd.Series(
                {
                    'maxmem': x[self.reader_cls.MaxRssFloat].max(),
                    'maxtime': x[self.reader_cls.DeltaMinutes].max(),
                    'n_obs': x['n_obs'].max(),
                    'n_perts': x['n_perts'].max()
                }
            )
        ).reset_index()
        return agg_data
    
    def save(self, out_dir: str, mode: list[str] = ['data', 'agg_data']) -> None:
        if 'data' in mode:
            o = os.path.join(out_dir, 'trace_data.csv')
            self.data.to_csv(o)
        if 'agg_data' in mode:
            o = os.path.join(out_dir, 'trace_agg_data.csv')
            self.agg_data.to_csv(o)
