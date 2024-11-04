from time import perf_counter
import numpy as np
import bayesflow as bf


def eval_performance(conf, amortizer, n_samples=5000, scale=False, **kwargs):
    
    perf = {}
    
    # Get samples and measure time
    t1 = perf_counter()
    samples = amortizer.sample(conf, n_samples=n_samples, to_numpy=False, **kwargs)
    t2 = perf_counter()
    perf['time'] = t2 - t1
    
    if type(samples) is not np.ndarray:
        samples = samples.numpy()
    
    params = conf['parameters']
    
    # Calibration
    cal = bf.computational_utilities.posterior_calibration_error(samples, params)
    perf['MaxECE'] = np.max(cal)
    perf['MeanECE'] = np.mean(cal)
    perf['MedianECE'] = np.median(cal)
    perf['MinECE'] = np.min(cal)
    
    if scale is not None:
        samples = (samples - scale[0]) / scale[1]
        params = (params - scale[0]) / scale[1]
    
    # R2
    r2 = np.mean(np.sqrt((params[:, np.newaxis, :] - samples)**2), axis=1)
    perf['RMSE'] = np.mean(r2)
    return perf
