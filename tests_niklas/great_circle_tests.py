import numpy as np

diff = lambda a, b : a-b
norm = lambda a, b : np.abs(diff(a,b))

funcsdict = {"diff": diff,
             "norm": norm}

#compute latitudes for the great-circle path starting at the zero-meridian and equator, with longitude as independent variable
def great_circle_path(theta, lon):
#    a = np.sin(lon)/np.sin(theta)
    return np.arctan(np.tan(theta)*np.sin(lon))

def compute_deviation(lats_comp, lons_comp, theta, func="diff"):
    if not func in funcsdict.keys():
        print("method not found, reverting back to differences")
        method = funcsdict["diff"]
    else:
        method = funcsdict[func]

    lats_ana = great_circle_path(theta, lons_comp)
    return method(lats_ana, lats_comp)

#function that returns the numerically integrated error
def integrated_deviation(lats_comp, lons_comp, theta, dt, func="norm"):
    vals = compute_deviation(lats_comp, lons_comp, theta, func="norm")
    return np.sum(vals)/dt
