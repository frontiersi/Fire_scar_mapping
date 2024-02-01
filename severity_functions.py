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

def dist_severity(params):
    """
    multiprocess version with shared memory of the severity algorithm
    """

    nbr = np.frombuffer(shared_in_arr01.get_obj(), dtype=np.float32).reshape(
        (-1, params[2])
    )
    nbr_dist = np.frombuffer(shared_in_arr02.get_obj(), dtype=np.float32).reshape(
        (-1, params[2])
    )
    c_dist = np.frombuffer(shared_in_arr03.get_obj(), dtype=np.float32).reshape(
        (-1, params[2])
    )
    change_dir = np.frombuffer(shared_in_arr04.get_obj(), dtype=np.int16).reshape(
        (-1, params[2])
    )
    nbr_outlier = np.frombuffer(shared_in_arr05.get_obj(), dtype=np.float32)
    cdist_outlier = np.frombuffer(shared_in_arr06.get_obj(), dtype=np.float32)
    t = np.frombuffer(shared_in_arr07.get_obj(), dtype=np.float64)

    sev = np.frombuffer(shared_out_arr01.get_obj(), dtype=np.float64)
    dates = np.frombuffer(shared_out_arr02.get_obj(), dtype=np.float64)
    days = np.frombuffer(shared_out_arr03.get_obj(), dtype=np.float64)

    for i in range(params[0], params[1]):
        sev[i], dates[i], days[i] = severity(
            nbr[:, i],
            nbr_dist[:, i],
            c_dist[:, i],
            change_dir[:, i],
            nbr_outlier[i],
            cdist_outlier[i],
            t,
            method=params[3],
        )

def severity(
    nbr,
    nbr_dist,
    cos_dist,
    change_dir,
    nbr_outlier,
    cos_dist_outlier,
    t,
    method="NBRdist",
):
    """
    Returns the severity,duration and start date of the change.
    Args:
        nbr: normalised burn ratio in tx1 dimension
        nbr_dist: nbr distance in tx1 dimension
        cos_dist: cosine distance in tx1 dimension
        change_dir: NBR change direction in tx1 dimension
        nbr_outlier: outlier values for NBRdist
        cos_dist_outlier: outlier values for CDist
        t: dates of observations
        data: xarray including the cosine distances, NBR distances, NBR, change direction and outliers value
        method: two options to choose
            NBR: use cosine distance together with NBR<0
            NBRdist: use both cosine distance, NBR euclidean distance, and NBR change direction for change detection

    Returns:
        sevindex: severity
        startdate: first date change was detected
        duration: duration between the first and last date the change exceeded the outlier threshold
    """

    sevindex = 0
    startdate = 0
    duration = 0

    notnanind = np.where(~np.isnan(cos_dist))[0]  # remove the nan values for each pixel

    if method == "NBR":  # cosdist above the line and NBR<0
        outlierind = np.where(
            (cos_dist[notnanind] > cos_dist_outlier) & (nbr[notnanind] < 0)
        )[0]
        cosdist = cos_dist[notnanind]

    elif (
        method == "NBRdist"
    ):  # both cosdist and NBR dist above the line and it is negative change
        outlierind = np.where(
            (cos_dist[notnanind] > cos_dist_outlier)
            & (nbr_dist[notnanind] > nbr_outlier)
            & (change_dir[notnanind] == 1)
        )[0]

        cosdist = cos_dist[notnanind]
    else:
        raise ValueError
    t = t.astype("datetime64[ns]")
    t = t[notnanind]
    outlierdates = t[outlierind]
    n_out = len(outlierind)
    area_above_d0 = 0
    if n_out >= 2:
        tt = []
        for ii in range(0, n_out):
            if outlierind[ii] + 1 < len(t):
                u = np.where(t[outlierind[ii] + 1] == outlierdates)[
                    0
                ]  # next day have to be outlier to be included
                # print(u)

                if len(u) > 0:
                    t1_t0 = (
                        (t[outlierind[ii] + 1] - t[outlierind[ii]])
                        / np.timedelta64(1, "s")
                        / (60 * 60 * 24)
                    )
                    y1_y0 = (
                        cosdist[outlierind[ii] + 1] + cosdist[outlierind[ii]]
                    ) - 2 * cos_dist_outlier
                    area_above_d0 = (
                        area_above_d0 + 0.5 * y1_y0 * t1_t0
                    )  # calculate the area under the curve
                    duration = duration + t1_t0
                    tt.append(ii)  # record the index where it is detected as a change

        if len(tt) > 0:
            startdate = t[outlierind[tt[0]]]  # record the date of the first change
            sevindex = area_above_d0

    return sevindex, startdate, duration

def outline_to_mask(line, x, y):
    """Create mask from outline contour

    Parameters
    ----------
    line: array-like (N, 2)
    x, y: 1-D grid coordinates (input for meshgrid)

    Returns
    -------
    mask : 2-D boolean array (True inside)

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> poly = Point(0, 0).buffer(1)
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.linspace(-5, 5, 100)
    >>> mask = outline_to_mask(poly.boundary, x, y)
    """
    import matplotlib.path as mplp

    mpath = mplp.Path(line)
    x_val, y_val = np.meshgrid(x, y)
    points = np.array((x_val.flatten(), y_val.flatten())).T
    mask = mpath.contains_points(points).reshape(x_val.shape)

    return mask

def post_filtering(sev, hotspots_filtering=False, date_filtering=True):
    """
    This function cleans up the potential cloud contaminated results with hotspots data and start date
    variables:
        sev: outputs from BurnCube
        hotspots_filtering: whether filtering the results with hotspots data
        date_filtering: whether filtering the results with only five major changes with startdate info
    outputs:
        sev: with one extra layer 'cleaned'
    """
    if "Moderate" in sev.keys():
        burn_pixel = (
            sev.Moderate
        ) 
        filtered_burnscar = np.zeros(burn_pixel.data.shape).astype("f4")

        if hotspots_filtering:

            all_labels = measure.label(burn_pixel.data, background=0)

            if ("Corroborate" in sev.keys()) * (sev.Corroborate.data.sum() > 0):
                hs_pixel = sev.Corroborate  
                # tmp = all_labels * hs_pixel.data.astype("int32")
                tmp = all_labels * hs_pixel.data.astype(np.int32)
                overlappix = (-hs_pixel.data + burn_pixel.data * 2).reshape(-1)
                if len(overlappix[overlappix == 2]) > 0:
                    overlaplabels = np.unique(tmp)
                    labels = overlaplabels[overlaplabels > 0]
                    for i in labels:
                        seg = np.zeros(burn_pixel.data.shape)
                        seg[all_labels == i] = 1
                        if np.sum(seg * hs_pixel.data) > 0:
                            filtered_burnscar[seg == 1] = 1
                else:
                    filtered_burnscar[:] = burn_pixel.data.copy()

            else:
                filtered_burnscar = np.zeros(burn_pixel.data.shape)

            cleaned = np.zeros(burn_pixel.data.shape)
            filtered_burnscar[filtered_burnscar == 0] = np.nan
            clean_date = filtered_burnscar * sev.StartDate
            mask = np.where(~np.isnan(clean_date.data))
            clean_date = clean_date.astype("datetime64[ns]")
            cleaned[mask[0], mask[1]] = pd.DatetimeIndex(
                clean_date.data[mask[0], mask[1]]
            ).month
            # sev["Cleaned"] = (("y", "x"), cleaned.astype("int16"))
            sev["Cleaned"] = (("y", "x"), cleaned.astype(np.int16))

        if date_filtering:
            # hotspotsmask = burn_pixel.data.copy().astype("float32")
            hotspotsmask = burn_pixel.data.copy().astype(np.float32)
            hotspotsmask[hotspotsmask == 0] = np.nan
            firedates = (sev.StartDate.data * hotspotsmask).reshape(-1)
            values, counts = np.unique(
                firedates[~np.isnan(firedates)], return_counts=True
            )
            sortcounts = np.array(sorted(counts, reverse=True))
            datemask = np.zeros(sev.StartDate.data.shape)
            hp_masked_date = sev.StartDate * hotspotsmask.copy()
            # print(len(sortcounts))
            if len(sortcounts) <= 2:
                fireevents = sortcounts[0:1]
            else:
                fireevents = sortcounts[0:5]

            for fire in fireevents:
                # print('Change detected at: ',values[counts==fire].astype('datetime64[ns]')[0])
                firedate = values[counts == fire]

                for firei in firedate:
                    start = (
                        firei.astype("datetime64[ns]") - np.datetime64(1, "M")
                    ).astype("datetime64[ns]")
                    end = (
                        firei.astype("datetime64[ns]") - np.datetime64(-1, "M")
                    ).astype("datetime64[ns]")

                    row, col = np.where(
                        (hp_masked_date.data.astype("datetime64[ns]") >= start)
                        & (hp_masked_date.data.astype("datetime64[ns]") <= end)
                    )

                    datemask[row, col] = 1

            # burn_pixel.data = burn_pixel.data*datemask
            # filtered_burnscar = burn_pixel.data.astype("float32").copy()
            filtered_burnscar = burn_pixel.data.astype(np.float32).copy()
            filtered_burnscar = filtered_burnscar * datemask
            filtered_burnscar[filtered_burnscar == 0] = np.nan
            cleaned = np.zeros(burn_pixel.data.shape)
            clean_date = filtered_burnscar * sev.StartDate.data
            mask = np.where(~np.isnan(clean_date))
            clean_date = clean_date.astype("datetime64[ns]")
            cleaned[mask[0], mask[1]] = pd.DatetimeIndex(
                clean_date[mask[0], mask[1]]
            ).month
            # sev["Cleaned"] = (("y", "x"), cleaned.astype("int16"))
            sev["Cleaned"] = (("y", "x"), cleaned.astype(np.int16))

    return sev

def severitymapping(
    dists,
    outlrs,
    period,
    hotspotfile,
    method="NBR",
    growing=True,
    hotspots_period=None,
    n_procs=1,
):
    """Calculates burnt area for a given period
    Args:
        period: period of time with burn mapping interest,  e.g.('2015-01-01','2015-12-31')
        n_procs: tolerance criterion to stop iteration
        method: methods for change detection
        growing: whether to grow the region
    """

    if dists is None:
        logger.warning("no data available for severity mapping")
        return None

    c_dist = dists.CDist.data.reshape((len(dists.time), -1))
    cdist_outlier = outlrs.CDistoutlier.data.reshape(len(dists.x) * len(dists.y))
    nbr_dist = dists.NBRDist.data.reshape((len(dists.time), -1))
    nbr = dists.NBR.data.reshape((len(dists.time), -1))
    nbr_outlier = outlrs.NBRoutlier.data.reshape(len(dists.x) * len(dists.y))
    change_dir = dists.ChangeDir.data.reshape((len(dists.time), -1))

    if method == "NBR":
        tmp = (
            dists.CDist.where((dists.CDist > outlrs.CDistoutlier) & (dists.NBR < 0))
            .sum(axis=0)
            .data
        )
        tmp = tmp.reshape(len(dists.x) * len(dists.y))
        outlierind = np.where(tmp > 0)[0]

    elif method == "NBRdist":
        tmp = (
            dists.CDist.where(
                (dists.CDist > outlrs.CDistoutlier)
                & (dists.NBRDist > outlrs.NBRoutlier)
                & (dists.ChangeDir == 1)
            )
            .sum(axis=0)
            .data
        )
        tmp = tmp.reshape(len(dists.x) * len(dists.y))
        outlierind = np.where(tmp > 0)[0]

    else:
        raise ValueError

    outlierind.compute_chunk_sizes()
    if len(outlierind) == 0:
        logger.warning("no burnt area detected")
        return None
    # input shared arrays
    in_arr1 = mp.Array(ctypes.c_float, len(dists.time[:]) * len(outlierind))
    nbr_shared = np.frombuffer(in_arr1.get_obj(), dtype=np.float32).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    nbr_shared[:] = nbr[:, outlierind]

    in_arr2 = mp.Array(ctypes.c_float, len(dists.time[:]) * len(outlierind))
    nbr_dist_shared = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    nbr_dist_shared[:] = nbr_dist[:, outlierind]

    in_arr3 = mp.Array(ctypes.c_float, len(dists.time[:]) * len(outlierind))
    cosdist_shared = np.frombuffer(in_arr3.get_obj(), dtype=np.float32).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    cosdist_shared[:] = c_dist[:, outlierind]

    in_arr4 = mp.Array(ctypes.c_short, len(dists.time[:]) * len(outlierind))
    change_dir_shared = np.frombuffer(in_arr4.get_obj(), dtype=np.int16).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    change_dir_shared[:] = change_dir[:, outlierind]

    in_arr5 = mp.Array(ctypes.c_float, len(outlierind))
    nbr_outlier_shared = np.frombuffer(in_arr5.get_obj(), dtype=np.float32)
    nbr_outlier_shared[:] = nbr_outlier[outlierind]

    in_arr6 = mp.Array(ctypes.c_float, len(outlierind))
    cdist_outlier_shared = np.frombuffer(in_arr6.get_obj(), dtype=np.float32)
    cdist_outlier_shared[:] = cdist_outlier[outlierind]

    in_arr7 = mp.Array(ctypes.c_double, len(dists.time[:]))
    t = np.frombuffer(in_arr7.get_obj(), dtype=np.float64)
    # t[:] = dists.time.data.astype("float64")
    t[:] = dists.time.data.astype(np.float64)

    # output shared arrays
    out_arr1 = mp.Array(ctypes.c_double, len(outlierind))
    sev = np.frombuffer(out_arr1.get_obj(), dtype=np.float64)
    sev.fill(np.nan)

    out_arr2 = mp.Array(ctypes.c_double, len(outlierind))
    dates = np.frombuffer(out_arr2.get_obj(), dtype=np.float64)
    dates.fill(np.nan)

    out_arr3 = mp.Array(ctypes.c_double, len(outlierind))
    days = np.frombuffer(out_arr3.get_obj(), dtype=np.float64)
    days.fill(0)

    def init(
        shared_in_arr1_,
        shared_in_arr2_,
        shared_in_arr3_,
        shared_in_arr4_,
        shared_in_arr5_,
        shared_in_arr6_,
        shared_in_arr7_,
        shared_out_arr1_,
        shared_out_arr2_,
        shared_out_arr3_,
    ):
        global shared_in_arr01
        global shared_in_arr02
        global shared_in_arr03
        global shared_in_arr04
        global shared_in_arr05
        global shared_in_arr06
        global shared_in_arr07
        global shared_out_arr01
        global shared_out_arr02
        global shared_out_arr03

        shared_in_arr01 = shared_in_arr1_
        shared_in_arr02 = shared_in_arr2_
        shared_in_arr03 = shared_in_arr3_
        shared_in_arr04 = shared_in_arr4_
        shared_in_arr05 = shared_in_arr5_
        shared_in_arr06 = shared_in_arr6_
        shared_in_arr07 = shared_in_arr7_
        shared_out_arr01 = shared_out_arr1_
        shared_out_arr02 = shared_out_arr2_
        shared_out_arr03 = shared_out_arr3_

    with closing(
        mp.Pool(
            initializer=init,
            initargs=(
                in_arr1,
                in_arr2,
                in_arr3,
                in_arr4,
                in_arr5,
                in_arr6,
                in_arr7,
                out_arr1,
                out_arr2,
                out_arr3,
            ),
            processes=n_procs,
        )
    ) as p:
        chunk = 1
        if len(outlierind) == 0:
            return
        p.map_async(
            dist_severity,
            [
                (i, min(len(outlierind), i + chunk), len(outlierind), method)
                for i in range(0, len(outlierind), chunk)
            ],
        )

    p.join()

    sevindex = np.zeros(len(dists.y) * len(dists.x))
    duration = np.zeros(len(dists.y) * len(dists.x)) * np.nan
    startdate = np.zeros(len(dists.y) * len(dists.x)) * np.nan
    sevindex[outlierind] = sev
    duration[outlierind] = days
    startdate[outlierind] = dates
    sevindex = sevindex.reshape((len(dists.y), len(dists.x)))
    duration = duration.reshape((len(dists.y), len(dists.x)))
    startdate = startdate.reshape((len(dists.y), len(dists.x)))
    startdate[startdate == 0] = np.nan
    duration[duration == 0] = np.nan
    del (
        in_arr1,
        in_arr2,
        in_arr3,
        in_arr4,
        in_arr5,
        in_arr6,
        out_arr1,
        out_arr2,
        out_arr3,
    )
    del (
        sev,
        days,
        dates,
        nbr_shared,
        nbr_dist_shared,
        cosdist_shared,
        nbr_outlier_shared,
        change_dir_shared,
        cdist_outlier_shared,
    )

    out = xr.Dataset(coords={"y": dists.y[:], "x": dists.x[:]})
    out["StartDate"] = (("y", "x"), startdate)
    # out["Duration"] = (("y", "x"), duration.astype("int16"))
    out["Duration"] = (("y", "x"), duration.astype(np.int16))
    burnt = np.zeros((len(dists.y), len(dists.x)))
    burnt[duration > 1] = 1
    # out["Severity"] = (("y", "x"), sevindex.astype("float32"))
    out["Severity"] = (("y", "x"), sevindex.astype(np.float32))
    # out["Severe"] = (("y", "x"), burnt.astype("int16"))
    out["Severe"] = (("y", "x"), burnt.astype(np.int16))

    # count = dists["NBR"].count(dim="time").astype("int16")
    count = dists["NBR"].count(dim="time").astype(np.int16)
    out["Count"] = count

    if growing:
        burn_area, growing_dates = region_growing(out, dists, outlrs)
        # out["Moderate"] = (("y", "x"), burn_area.astype("int16"))
        out["Moderate"] = (("y", "x"), burn_area.astype(np.int16))
        growing_dates[growing_dates == 0] = np.nan
        out["StartDate"] = (("y", "x"), growing_dates)

    extent = [
        np.min(dists.x.data),
        np.max(dists.x.data),
        np.min(dists.y.data),
        np.max(dists.y.data),
    ]

    if hotspots_period is None:
        hotspots_period = period
        polygons=None
    else:
        polygons = hotspot_polygon(
            hotspots_period, extent, 4000, hotspotfile
        )  # generate hotspot polygons with 4km buffer

    # default mask
    hot_spot_mask = np.zeros((len(dists.y), len(dists.x)))

    if polygons is None or polygons.is_empty:
        logger.warning("no hotspots data")
    else:
        coords = out.coords

        if polygons.type == "MultiPolygon":
            for polygon in polygons.geoms:
                hot_spot_mask_tmp = outline_to_mask(
                    polygon.exterior.coords, coords["x"], coords["y"]
                )
                hot_spot_mask = hot_spot_mask_tmp + hot_spot_mask
            hot_spot_mask = xr.DataArray(hot_spot_mask, coords=coords, dims=("y", "x"))
        if polygons.type == "Polygon":
            hot_spot_mask = outline_to_mask(polygons.exterior, coords["x"], coords["y"])
            hot_spot_mask = xr.DataArray(hot_spot_mask, coords=coords, dims=("y", "x"))

    # out["Corroborate"] = (("y", "x"), hot_spot_mask.data.astype("int16"))
    # out["Corroborate"] = (("y", "x"), hot_spot_mask.data.astype(np.int16))
    out["Corroborate"] = (("y", "x"), hot_spot_mask.data)
    out = post_filtering(out, hotspots_filtering=False, date_filtering=False)
    return out

def region_growing(severity, dists, outlrs):
    """
    The function includes further areas that do not qualify as outliers but do show a substantial decrease in NBR and
    are adjoining pixels detected as burns. These pixels are classified as 'moderate severity burns'.
        Note: this function is build inside the 'severity' function
            Args:
                severity: xarray including severity and start-date of the fire
    """
    start_date = severity.StartDate.data[~np.isnan(severity.StartDate.data)].astype(
        "datetime64[ns]"
    )
    change_dates = np.unique(start_date)
    z_distance = 2 / 3  # times outlier distance (eq. 3 stdev)

    # see http://www.scipy-lectures.org/packages/scikit-image/index.html#binary-segmentation-foreground-background
    fraction_seedmap = 0.25  # this much of region must already have been mapped as burnt to be included
    seed_map = (severity.Severe.data > 0).astype(
        int
    )  # use 'Severe' burns as seed map to grow

    start_dates = np.zeros((len(dists.y), len(dists.x)))
    start_dates[~np.isnan(severity.StartDate.data)] = start_date
    tmp_map = seed_map
    annual_map = seed_map
    # grow the region based on StartDate
    for d in change_dates:

        di = str(d)[:10]
        ti = np.where(dists.time > np.datetime64(di))[0][0]
        nbr_score = (dists.ChangeDir * dists.NBRDist)[ti, :, :] / outlrs.NBRoutlier
        cos_score = (dists.ChangeDir * dists.CDist)[ti, :, :] / outlrs.CDistoutlier
        potential = ((nbr_score > z_distance) & (cos_score > z_distance)).astype(int)
        # Use the following line if using NBR is preferred
        # Potential = ((dists.NBR[ti, :, :] > 0) & (cos_score > z_distance)).astype(int)

        all_labels = measure.label(
            potential.astype(int).values, background=0
        )  # labelled all the conneted component
        new_potential = ndimage.mean(seed_map, labels=all_labels, index=all_labels)
        new_potential[all_labels == 0] = 0

        annual_map = annual_map + (new_potential > fraction_seedmap).astype(int)
        annual_map = (annual_map > 0).astype(int)
        start_dates[
            (annual_map - tmp_map) > 0
        ] = d  # assign the same date to the growth region
        tmp_map = annual_map

    burn_extent = annual_map

    return burn_extent, start_dates

def hotspot_polygon(period, extent, buffersize, hotspotfile):
    """Create polygons for the hotspot with a buffer
    year: given year for hotspots data
    extent: [xmin,xmax,ymin,ymax] in crs EPSG:3577
    buffersize: in meters

    Examples:
    ------------
    >>>year=2017
    >>>extent = [1648837.5, 1675812.5, -3671837.5, -3640887.5]
    >>>polygons = hotspot_polygon(year,extent,4000)
    """

    # print("extent", extent)

    # year = int(str(period[0])[0:4])
    # if year >= 2019:
    #    logger.warning("No complete hotspots data after 2018")
    #    return None

    _ = s3fs.S3FileSystem(anon=True)

    # hotspotfile = (
    #    "s3://dea-public-data-dev/projects/burn_cube/ancillary_file/hotspot_historic.csv"
    # )

    # if os.path.isfile(hotspotfile):
    #    column_names = ["datetime", "sensor", "latitude", "longitude"]
    #    table = pd.read_csv(hotspotfile, usecols=column_names)
    # else:
    #    logger.warning("No hotspots file is found")
    #    return None

    column_names = ["datetime", "sensor", "latitude", "longitude"]
    table = pd.read_csv(hotspotfile, usecols=column_names)

    start = (
        np.datetime64(period[0]).astype("datetime64[ns]") - np.datetime64(2, "M")
    ).astype("datetime64[ns]")
    stop = np.datetime64(period[1])
    extent[0] = extent[0] - 100000
    extent[1] = extent[1] + 100000
    extent[2] = extent[2] - 100000
    extent[3] = extent[3] + 100000

    dates = pd.to_datetime(table.datetime.apply(lambda x: x.split("+")[0]).values)

    transformer = pyproj.Transformer.from_crs("EPSG:3577", "EPSG:4283")
    lat, lon = transformer.transform(extent[0:2], extent[2:4])

    index = np.where(
        (table.sensor == "MODIS")
        & (dates >= start)
        & (dates <= stop)
        & (table.latitude <= lat[1])
        & (table.latitude >= lat[0])
        & (table.longitude <= lon[1])
        & (table.longitude >= lon[0])
    )[0]

    latitude = table.latitude.values[index]
    longitude = table.longitude.values[index]

    reverse_transformer = pyproj.Transformer.from_crs("EPSG:4283", "EPSG:3577")
    easting, northing = reverse_transformer.transform(latitude, longitude)

    patch = [
        Point(easting[i], northing[i]).buffer(buffersize) for i in range(0, len(index))
    ]
    polygons = unary_union(patch)

    return polygons