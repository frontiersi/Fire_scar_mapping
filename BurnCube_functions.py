import ctypes
import logging
import multiprocessing as mp
from contextlib import closing

import numpy as np
import pandas as pd
import pyproj
import s3fs
import xarray as xr
from scipy import ndimage
from shapely import geometry
from shapely.geometry import Point
from shapely.ops import unary_union
from skimage import measure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def _zvalue_from_index(arr, ind):
    """
    private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    modified from https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    with order of nR and nC fixed.
    """
    # get number of columns and rows
    _, nr, nc = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nr * nc * ind + nc * np.arange(nr)[:, np.newaxis] + np.arange(nc)
    return np.take(arr, idx)

def nanpercentile(inarr, q):
    """
    faster nanpercentile than np.nanpercentile for axis 0 of a 3D array.
    modified from https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    """
    arr = inarr.copy()
    # valid (non NaN) observations along the first axis
    valid_obs = np.isfinite(arr).sum(axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr.sort(axis=0)

    # loop over requested quantiles
    if type(q) is list:
        qs = q
    else:
        qs = [q]
    quant_arrs = np.empty(shape=(len(qs), arr.shape[1], arr.shape[2]))
    quant_arrs.fill(np.nan)

    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr, f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr, c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr, f_arr)[fc_equal_k_mask]

        quant_arrs[i] = quant_arr

    if quant_arrs.shape[0] == 1:
        return np.squeeze(quant_arrs, axis=0)
    else:
        return quant_arrs

def outliers(dataset, distances):
    """
    Calculate the outliers for distances for change detection
    """
    logger.info("begin to process outlier")

    if distances is None:
        logger.warning("no distances for the outlier calculations")
        return
    nbr_ps = nanpercentile(distances.NBRDist.data, [25, 75])

    nbr_outlier = nbr_ps[1] + 1.5 * (nbr_ps[1] - nbr_ps[0])
    cos_distps = nanpercentile(distances.CDist.data, [25, 75])
    cos_dist_outlier = cos_distps[1] + 1.5 * (cos_distps[1] - cos_distps[0])

    ds = xr.Dataset(
        coords={"y": dataset.y[:], "x": dataset.x[:]}, attrs={"crs": "EPSG:3577"}
    )
    ds["CDistoutlier"] = (("y", "x"), cos_dist_outlier.astype("float32"))
    ds["NBRoutlier"] = (("y", "x"), nbr_outlier.astype("float32"))
    return ds


def distances(ard, geomed, n_procs=1):
    """
    Calculates the cosine distance between observation and reference.
    The calculation is point based, easily adaptable to any dimension.
        Note:
            This method saves the result of the computation into the
            dists variable: p-dimensional vector with geometric
            median reflectances, where p is the number of bands.
        Args:
            ard: load from ODC
            geomed: load from odc-stats GeoMAD plugin result
            n_procs: tolerance criterion to stop iteration
    """

    n = len(ard.y) * len(ard.x)
    _x = ard

    t_dim = _x.time.data
    if len(t_dim) < 1:
        logger.warning(f"--- {len(t_dim)} observations")
        return

    # measurements = ['nbart_blue', 'nbart_green', 'nbart_red','nbart_nir_1', 'nbart_swir_3']
    # nir = _x[3, :, :, :].data.astype("float32")
    # swir2 = _x[4, :, :, :].data.astype("float32") 
    nir = _x[3, :, :, :].data.astype(np.float32)
    swir2 = _x[4, :, :, :].data.astype(np.float32)

    nir[nir <= 0] = np.nan
    swir2[swir2 <= 0] = np.nan
    nbr = (nir - swir2) / (nir + swir2)

    out_arr1 = mp.Array(ctypes.c_float, len(t_dim) * n)
    out_arr2 = mp.Array(ctypes.c_float, len(t_dim) * n)
    out_arr3 = mp.Array(ctypes.c_short, len(t_dim) * n)

    cos_dist = np.frombuffer(out_arr1.get_obj(), dtype=np.float32).reshape((len(t_dim), n))
    cos_dist.fill(np.nan)
    nbr_dist = np.frombuffer(out_arr2.get_obj(), dtype=np.float32).reshape((len(t_dim), n))
    nbr_dist.fill(np.nan)
    direction = np.frombuffer(out_arr3.get_obj(), dtype=np.int16).reshape((len(t_dim), n))
    direction.fill(0)
    
    in_arr1 = mp.Array(ctypes.c_short, len(ard.variable) * len(_x.time) * n)
    x = np.frombuffer(in_arr1.get_obj(), dtype=np.int16).reshape((len(ard.variable), len(_x.time), n))
    x[:] = _x.data.reshape(len(ard.variable), len(_x.time), -1)

    in_arr2 = mp.Array(ctypes.c_float, len(ard.variable) * n)
    gmed = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape((len(ard.variable), n))
    gmed[:] = geomed.data.reshape(len(ard.variable), -1)

#     in_arr1 = mp.Array(ctypes.c_short, len(ard.band) * len(_x.time) * n)
#     x = np.frombuffer(in_arr1.get_obj(), dtype=np.int16).reshape((len(ard.band), len(_x.time), n))
#     x[:] = _x.data.reshape(len(ard.band), len(_x.time), -1)

#     in_arr2 = mp.Array(ctypes.c_float, len(ard.band) * n)
#     gmed = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape((len(ard.band), n))
#     gmed[:] = geomed.data.reshape(len(ard.band), -1)

    def init(shared_in_arr1_,
        shared_in_arr2_,
        shared_out_arr1_,
        shared_out_arr2_,
        shared_out_arr3_,):
        global shared_in_arr1
        global shared_in_arr2
        global shared_out_arr1
        global shared_out_arr2
        global shared_out_arr3

        shared_in_arr1 = shared_in_arr1_
        shared_in_arr2 = shared_in_arr2_
        shared_out_arr1 = shared_out_arr1_
        shared_out_arr2 = shared_out_arr2_
        shared_out_arr3 = shared_out_arr3_

    with closing(
        mp.Pool(
            initializer=init,
            initargs=(
                in_arr1,
                in_arr2,
                out_arr1,
                out_arr2,
                out_arr3,),processes=n_procs,)) as p:
        chunk = 1
        if n == 0:
            logger.warning("no point")
            return
        p.map_async(
            dist_distance,
            [(i, min(n, i + chunk), x.shape) for i in range(0, n, chunk)],)

    p.join()

    ds = xr.Dataset(
        coords={
            "time": t_dim,
            "y": ard.y[:],
            "x": ard.x[:],
            "band": ard.variable,},
            # "band": ard.band,},
        attrs={"crs": "EPSG:3577"},)

    ds["CDist"] = (
        ("time", "y", "x"),
        # cos_dist[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype("float32"),
        cos_dist[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype(np.float32),
    )
    ds["NBRDist"] = (
        ("time", "y", "x"),
        nbr_dist[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype(np.float32),
    )
    ds["ChangeDir"] = (
        ("time", "y", "x"),
        direction[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype(np.float32),
    )
    ds["NBR"] = (("time", "y", "x"), nbr)

    del (in_arr1,in_arr2,out_arr1,out_arr2,out_arr3,gmed,ard,cos_dist,nbr_dist,direction,nbr,)

    return ds

def dist_distance(params):
    """
    multiprocess version with shared memory of the cosine distances and nbr distances
    """
    ard = np.frombuffer(shared_in_arr1.get_obj(), dtype=np.int16).reshape(params[2])
    gmed = np.frombuffer(shared_in_arr2.get_obj(), dtype=np.float32).reshape(
        (params[2][0], params[2][2])
    )
    cos_dist = np.frombuffer(shared_out_arr1.get_obj(), dtype=np.float32).reshape(
        (params[2][1], params[2][2])
    )
    nbr_dist = np.frombuffer(shared_out_arr2.get_obj(), dtype=np.float32).reshape(
        (params[2][1], params[2][2])
    )
    direction = np.frombuffer(shared_out_arr3.get_obj(), dtype=np.int16).reshape(
        (params[2][1], params[2][2])
    )

    for i in range(params[0], params[1]):
        ind = np.where(ard[1, :, i] > 0)[0]  # 7 bands setting
        # ind = np.where(ard[0, :, i] > 0)[0] # 3 bands setting

        if len(ind) > 0:
            cos_dist[ind, i] = cos_distance(gmed[:, i], ard[:, ind, i])
            # # 7 bands setting
            # nbrmed = (gmed[3, i] - gmed[5, i]) / (gmed[3, i] + gmed[5, i])
            # nbr = (ard[3, :, i] - ard[5, :, i]) / (ard[3, :, i] + ard[5, :, i])
            
            # measurements = ['nbart_blue', 'nbart_green', 'nbart_red','nbart_nir_1', 'nbart_swir_3']
            nbrmed = (gmed[3, i] - gmed[4, i]) / (gmed[3, i] + gmed[4, i])
            nbr = (ard[3, :, i] - ard[4, :, i]) / (ard[3, :, i] + ard[4, :, i])

            # 3 bands setting
            # nbrmed = (gmed[1, i] - gmed[2, i]) / (gmed[1, i] + gmed[2, i])
            # nbr = (ard[1, :, i] - ard[2, :, i]) / (ard[1, :, i] + ard[2, :, i])
            nbr_dist[ind, i], direction[ind, i] = nbr_eucdistance(nbrmed, nbr[ind])
            
def nbr_eucdistance(ref, obs):
    """
    Returns the euclidean distance between the NBR at each time step with the NBR calculated from the geometric medians
    and also the direction of change to the NBR from the geometric medians.

    Args:
        ref: NBR calculated from geometric median, one value
        obs: NBR time series, 1-D time series array with ndays

    Returns:
        nbr_dist: the euclidean distance
        direction: change direction (1: decrease; 0: increase) at each time step in [ndays]
    """
    nbr_dist = np.empty((obs.shape[0],))
    direction = np.zeros((obs.shape[0],), dtype="uint8")
    nbr_dist.fill(np.nan)
    index = np.where(~np.isnan(obs))[0]
    euc_dist = obs[index] - ref
    euc_norm = np.sqrt(euc_dist**2)
    nbr_dist[index] = euc_norm
    direction[index[euc_dist < -0.05]] = 1

    return nbr_dist, direction

def cos_distance(ref, obs):
    """
    Returns the cosine distance between observation and reference
    The calculation is point based, easily adaptable to any dimension.
    Args:
        ref: reference (1-D array with multiple bands) e.g., geomatrix median [Nbands]
        obs: observation (with multiple bands, e.g. 6) e.g.,  monthly geomatrix median or reflectance [Nbands,ndays]

    Returns:
        cosdist: the cosine distance at each time step in [ndays]
    """
    ref = ref.astype(np.float32)[:, np.newaxis]
    obs = obs.astype(np.float32)
    cosdist = np.empty((obs.shape[1],))
    cosdist.fill(np.nan)

    cosdist = np.transpose(
        1
        - np.nansum(ref * obs, axis=0)
        / np.sqrt(np.sum(ref**2))
        / np.sqrt(np.nansum(obs**2, axis=0))
    )
    return cosdist