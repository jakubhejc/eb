import os
import csv
import argparse
import numpy as np
import pandas as pd
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl


class Metrics():
    def __init__(self, smooth=None) -> None:
        self.smooth = smooth
        self.default_methods = ['sdnn']
        self.flag = False
        self.results = None       
    
    def smooth_series(self, numpy_series):
        temp = np.cumsum(numpy_series, dtype=float)
        temp[self.smooth:] = temp[self.smooth:] - temp[:-self.smooth]
        return temp[self.smooth-1:] / self.smooth

    def sdnn(self, numpy_series, n:int=1):
        """Standard deviation of NN intervals."""
        out = td.sdnn(numpy_series)
        return 'SDNN', out['sdnn']

    def sdsd(self, numpy_series):
        """Standard deviation of successive differences"""
        out = td.sdsd(numpy_series)
        return 'SDSD', out['sdsd']
    
    def rmssd(self, numpy_series):
        """Root mean square of successive differences"""
        out = td.rmssd(numpy_series)
        return 'RMSSD', out['rmssd']
    
    def psd(self, numpy_series):
        """Histogram triangular index"""
        PARAMS = ['fft_abs', 'fft_rel']        

        param_names = [
            'FFT_abs_low',
            'FFT_abs_mid',
            'FFT_abs_high',
            'FFT_rel_low',
            'FFT_rel_mid',
            'FFT_rel_high',
        ]

        out = fd.welch_psd(numpy_series, show=False)

        cum_out = list()
        for param in PARAMS:
            cum_out.extend(list(out[param]))
        
        return param_names, cum_out

    def poincare(self, numpy_series):
        """Histogram triangular index"""
        PARAMS = ['sd1', 'sd2', 'ellipse_area']

        param_names = [
            'SD1',
            'SD2',
            'Area',            
        ]

        out = nl.poincare(numpy_series, show=False)
        
        cum_out = list()
        for param in PARAMS:
            cum_out.append(out[param])
        
        return param_names, cum_out
        
    def sample_entropy(self, numpy_series):
        """Sample entropy"""
        out = nl.sample_entropy(numpy_series)
        return 'Entropy', out['sampen']

    def compute(self, numpy_series, method_list:list=None):
        if method_list is None:
            method_list = self.default_methods
        
        if self.smooth > 0:
            numpy_series = self.smooth_series(numpy_series)
        
        col_names = list()
        row_values = list()
        for method in method_list:            
            # function handler from attribute name
            fce = getattr(self, method)

            # evaluate function
            param_name, out = fce(numpy_series)

            if isinstance(out, list):
                col_names.extend(param_name)
                row_values.extend(out)
            else:
                col_names.append(param_name)
                row_values.append(out)
        
        df = pd.DataFrame(data=[row_values], columns=col_names)
        if not self.flag:
            self.flag = True
            self.results = df
        else:
            self.results = pd.concat([self.results, df], axis=0)