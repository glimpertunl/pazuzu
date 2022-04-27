from __future__ import print_function
import os
import operator
import netCDF4 as ncdf
import wrf
import numpy as np
import time
import cartopy.crs as crs
import cartopy.feature as feature
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import skimage
import scipy.signal as signal
import scipy.ndimage as ndi

#Includes functions that read in variables from a netcdf file (getwrfvar), read in the forecast from the ensemble (readensdata), calculate parameters of forecast (fcst_resp)
def getwrfvar (ncfile, varname, replacenan=0, preslevel=500, preslevel2=300, zagllevel=6, zagltop = 5000, zaglbottom = 2000):
#zagllevel = km, zagltop/bottom = m
	# 2-meter temperature
	if varname.lower() == "t2":
		inputvar = wrf.getvar(ncfile, "T2")
		gridvar = inputvar
	# 2-meter dewpoint
	elif varname.lower() == "td2":
		inputvar = wrf.getvar(ncfile, "td2", units="K")
		gridvar = inputvar
	# 10-meter u-wind, rotated to Earth coordinates
	elif varname.lower() == "u10":
		inputvar, inputvar2 = wrf.getvar(ncfile, "uvmet10")
		gridvar = inputvar
	# 10-meter v-wind, rotated to Earth coordinates
	elif varname.lower() == "v10":
		inputvar, inputvar2 = wrf.getvar(ncfile, "uvmet10")
		gridvar = inputvar2
	# 10-meter wind speed
	elif varname.lower() == "wspd10":
		inputvar, inputvar2 = wrf.getvar(ncfile, "wspd_wdir10")
		gridvar = inputvar
	# Composite reflectivity
	elif varname.lower() == "mdbz":
		inputvar = wrf.getvar(ncfile, "mdbz")
		gridvar = inputvar
	# temperature interpolated to a pressure level
	elif varname.lower() == "t_pres":
		inputvar = wrf.getvar(ncfile, "temp")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True, field_type="tk")
		gridvar = inputvar2[0,:,:]
	# dewpoint interpolated to a pressure level
	elif varname.lower() == "td_pres":
		inputvar = wrf.getvar(ncfile, "td")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True, field_type="tc")
		gridvar = inputvar2[0,:,:]
	# geopotential height interpolated to a pressure level
	elif varname.lower() == "geopt_pres":
		inputvar = wrf.getvar(ncfile, "geopt")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True, field_type="z")
		gridvar = inputvar2[0,:,:]
	# Uwind interpolated to a pressure level
	elif varname.lower() == "ua_pres":
		inputvar = wrf.getvar(ncfile, "ua")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True)
		gridvar = inputvar2[0,:,:]
	# Vwind interpolated to a pressure level
	elif varname.lower() == "va_pres":
		inputvar = wrf.getvar(ncfile, "va")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True)
		gridvar = inputvar2[0,:,:]
	# Wwind interpolated to a pressure level
	elif varname.lower() == "wa_pres":
		inputvar = wrf.getvar(ncfile, "wa")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True)
		gridvar = inputvar2[0,:,:]
	# Uwind, rotated to Earth coordinates, interpolated to a pressure level
	elif varname.lower() == "uamet_pres":
		inputvar, inputvar2 = wrf.getvar(ncfile, "uvmet")
		inputvar3 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True)
		gridvar = inputvar3[0,:,:]
	# Vwind, rotated to Earth coordinates, interpolated to a pressure level
	elif varname.lower() == "vamet_pres":
		inputvar, inputvar2 = wrf.getvar(ncfile, "uvmet")
		inputvar3 = wrf.vinterp(ncfile, inputvar2, "pressure", [preslevel], extrapolate=True, log_p=True)
		gridvar = inputvar3[0,:,:]
	# Wind speed interpolated to a pressure level
	elif varname.lower() == "wspd_pres":
		inputvar = wrf.getvar(ncfile, "ua")
		inputvar2 = wrf.getvar(ncfile, "va")
		inputvar3 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True)
		inputvar4 = wrf.vinterp(ncfile, inputvar2, "pressure", [preslevel], extrapolate=True, log_p=True)
		inputvar5 = np.sqrt(np.square(inputvar3[0,:,:])+np.square(inputvar4[0,:,:]))
		inputvar3[0,:,:] = inputvar5[:,:]
		inputvar3.attrs["description"] = "wind speed"
		gridvar = inputvar3[0,:,:]
	# Vertical wind shear on pressure levels
	elif varname.lower() == "wshr_pres":
		inputvar = wrf.getvar(ncfile, "ua")
		inputvar2 = wrf.getvar(ncfile, "va")
		inputvar3 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel, preslevel2], extrapolate=True, log_p=True)
		inputvar4 = wrf.vinterp(ncfile, inputvar2, "pressure", [preslevel,preslevel2], extrapolate=True, log_p=True)
		inputvar5 = np.sqrt(np.square(inputvar3[0,:,:]-inputvar3[1,:,:])+np.square(inputvar4[0,:,:]-inputvar4[1,:,:]))
		inputvar3[0,:,:] = inputvar5[:,:]
		inputvar3.attrs["description"] = "vertical wind shear"
		gridvar = inputvar3[0,:,:]
	# absolute vorticity interpolated to a pressure level
	elif varname.lower() == "avo_pres":
		inputvar = wrf.getvar(ncfile, "avo")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True)
		gridvar = inputvar2[0,:,:]
	# equivalent potential temperature interpolated to a pressure level
	elif varname.lower() == "theta_e_pres":
		inputvar = wrf.getvar(ncfile, "theta_e")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "pressure", [preslevel], extrapolate=True, log_p=True, field_type="eth")
		gridvar = inputvar2[0,:,:]
	# temperature interpolated to height above ground level
	elif varname.lower() == "t_zagl":
		inputvar = wrf.getvar(ncfile, "temp")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False, field_type="tk")
		gridvar = inputvar2[0,:,:]
	# dewpoint interpolated to height above ground level
	elif varname.lower() == "td_zagl":
		inputvar = wrf.getvar(ncfile, "td")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False, field_type="tc")
		gridvar = inputvar2[0,:,:]
	# geopotential height interpolated to height above ground level
	elif varname.lower() == "geopt_zagl":
		inputvar = wrf.getvar(ncfile, "geopt")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False, field_type="z")
		gridvar = inputvar2[0,:,:]
	# Uwind interpolated to height above ground level
	elif varname.lower() == "ua_zagl":
		inputvar = wrf.getvar(ncfile, "ua")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		gridvar = inputvar2[0,:,:]
	# Vwind interpolated to height above ground level
	elif varname.lower() == "va_zagl":
		inputvar = wrf.getvar(ncfile, "va")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		gridvar = inputvar2[0,:,:]
	# Wwind interpolated to height above ground level
	elif varname.lower() == "wa_zagl":
		inputvar = wrf.getvar(ncfile, "wa")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		gridvar = inputvar2[0,:,:]
	# Uwind, rotated to Earth coordinates, interpolated to height above ground level
	elif varname.lower() == "uamet_zagl":
		inputvar, inputvar2 = wrf.getvar(ncfile, "uvmet")
		inputvar3 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		gridvar = inputvar3[0,:,:]
	# Vwind, rotated to Earth coordinates, interpolated to height above ground level
	elif varname.lower() == "vamet_zagl":
		inputvar, inputvar2 = wrf.getvar(ncfile, "uvmet")
		inputvar3 = wrf.vinterp(ncfile, inputvar2, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		gridvar = inputvar3[0,:,:]
	# Wind speed interpolated to height above ground level
	elif varname.lower() == "wspd_zagl":
		inputvar = wrf.getvar(ncfile, "ua")
		inputvar2 = wrf.getvar(ncfile, "va")
		inputvar3 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		inputvar4 = wrf.vinterp(ncfile, inputvar2, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		inputvar5 = np.sqrt(np.square(inputvar3[0,:,:])+np.square(inputvar4[0,:,:]))
		inputvar3[0,:,:] = inputvar5[:,:]
		inputvar3.attrs["description"] = "wind speed"
		gridvar = inputvar3[0,:,:]
	# Wind shear on height above ground levels
	elif varname.lower() == "wshr_zagl":
		inputvar = wrf.getvar(ncfile, "ua")
		inputvar2 = wrf.getvar(ncfile, "va")
		inputvar3 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagltop/1000,zaglbottom/1000], extrapolate=True, log_p=False)
		inputvar4 = wrf.vinterp(ncfile, inputvar2, "ght_agl", [zagltop/1000,zaglbottom/1000], extrapolate=True, log_p=False)
		inputvar5 = np.sqrt(np.square(inputvar3[0,:,:]-inputvar3[1,:,:])+np.square(inputvar4[0,:,:]-inputvar4[1,:,:]))
		inputvar3[0,:,:] = inputvar5[:,:]
		inputvar3.attrs["description"] = "vertical wind shear"
		gridvar = inputvar3[0,:,:]
	# absolute vorticity interpolated to height above ground level
	elif varname.lower() == "avo_zagl":
		inputvar = wrf.getvar(ncfile, "avo")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False)
		gridvar = inputvar2[0,:,:]
	# equivalent potential temperature interpolated to height above ground level
	elif varname.lower() == "theta_e_zagl":
		inputvar = wrf.getvar(ncfile, "theta_e")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False, field_type="eth")
		gridvar = inputvar2[0,:,:]
	# potential temperature interpolated to height above ground level
	elif varname.lower() == "theta_zagl":
		inputvar = wrf.getvar(ncfile, "theta")
		inputvar2 = wrf.vinterp(ncfile, inputvar, "ght_agl", [zagllevel], extrapolate=True, log_p=False, field_type="th")
		gridvar = inputvar2[0,:,:]
	# M-CAPE
	elif varname.lower() == "mcape":
		inputvar1, inputvar2, inputvar3, inputvar4 = wrf.getvar(ncfile, "cape_2d")
		gridvar = inputvar1.fillna(replacenan)
	# M-CIN
	elif varname.lower() == "mcin":
		inputvar1, inputvar2, inputvar3, inputvar4 = wrf.getvar(ncfile, "cape_2d")
		gridvar = inputvar2.fillna(replacenan)
	# M-LCL
	elif varname.lower() == "mlcl":
		inputvar1, inputvar2, inputvar3, inputvar4 = wrf.getvar(ncfile, "cape_2d")
		gridvar = inputvar3.fillna(replacenan)
	# M-LFC
	elif varname.lower() == "mlfc":
		inputvar1, inputvar2, inputvar3, inputvar4 = wrf.getvar(ncfile, "cape_2d")
		gridvar = inputvar4.fillna(replacenan)
	# Sea Level Pressure
	elif varname.lower() == "slp":
		inputvar = wrf.getvar(ncfile, "slp", units="Pa")
		gridvar = inputvar
	# storm relative helicity, layer is 0 to agl level
	elif varname.lower() == "srh":
		inputvar = wrf.getvar(ncfile, "helicity", top = zagltop)
		gridvar = inputvar
	# updraft helicity
	elif varname.lower() == "uh_zagl":
		inputvar = wrf.getvar(ncfile, "updraft_helicity", top = zagltop, bottom = zaglbottom)
		gridvar = inputvar
	else:
		gridvar = []
	return gridvar

def readensdata (ens_dir, ens_mems, ens_time, ens_var, replacenan=0, preslevel=500, preslevel2=300, zagllevel=6, zagltop = 5000, zaglbottom = 2000, lat_bnds=(0,0), lon_bnds=(0,0), get_scl_data=False, def_fcst_resp_box=False, get_lat_lon=False):
	for ens_fcst in np.arange(0,ens_mems,1):
		print(ens_fcst)
		forecast_dir = ens_dir + '/ENS_MEM_' + str(ens_fcst + 1)
		dir_list = os.listdir(forecast_dir)
		count_file = len(dir_list)
		dir_list.sort()
		first_file = dir_list[0]
		first_file_timestr = first_file[11:30]
		last_file = dir_list[count_file-1]
		last_file_timestr = last_file[11:30]
		date_pattern = '%Y-%m-%d_%H:%M:%S'
		first_ts = int(time.mktime(time.strptime(first_file_timestr, date_pattern)))
		last_ts = int(time.mktime(time.strptime(last_file_timestr, date_pattern)))
		ncfile = ncdf.Dataset(forecast_dir + '/' + dir_list[0])
		start_date = ncfile.getncattr('SIMULATION_START_DATE')
		ncfile = ncdf.Dataset(forecast_dir + '/' + dir_list[ens_time])
		inputvar = getwrfvar(ncfile, ens_var, replacenan=replacenan, preslevel=preslevel, preslevel2=preslevel2, zagllevel=zagllevel, zagltop=zagltop, zaglbottom=zaglbottom)
		if ens_fcst == 0:	
			xsize_data = inputvar.shape[1]
			ysize_data = inputvar.shape[0]
			ens_data = np.zeros((ens_mems,ysize_data,xsize_data))
			proj_data = wrf.get_cartopy(inputvar)
			xlim_data = wrf.cartopy_xlim(inputvar)
			ylim_data = wrf.cartopy_ylim(inputvar)
			dx_data = ncfile.getncattr('DX')
			dy_data = ncfile.getncattr('DY')
			wprj_map_proj = ncfile.getncattr('MAP_PROJ')
			wprj_truelat1 = ncfile.getncattr('TRUELAT1')
			wprj_truelat2 = ncfile.getncattr('TRUELAT2')
			wprj_stand_lon = ncfile.getncattr('STAND_LON')
			wprj_known_x = 1
			wprj_known_y = 1
			wprj_ref_latlon = wrf.xy_to_ll(ncfile,wprj_known_x,wprj_known_y,meta=False)
			wprj_ref_lat = wprj_ref_latlon[0]
			wprj_ref_lon = wprj_ref_latlon[1]
			wprj_pole_lat = ncfile.getncattr('POLE_LAT')
			wprj_pole_lon = ncfile.getncattr('POLE_LON')
			if ((get_lat_lon == True) or (def_fcst_resp_box == True)):
				lats, lons = wrf.latlon_coords(inputvar)
				lat_data = wrf.to_np(lats)
				lon_data = wrf.to_np(lons)
			else:
				lat_data = []
				lon_data = []
			if (def_fcst_resp_box == True):
				fcst_resp_box_mask = np.all([(lat_data >= lat_bnds[0]),(lat_data <= lat_bnds[1]),(lon_data >= lon_bnds[0]),(lon_data <= lon_bnds[1])],axis=0)
			else:
				fcst_resp_box_mask = []
			if (get_scl_data == True):
				scale_data = wrf.extract_vars(ncfile,0,["MAPFAC_MX","MAPFAC_MY"])
				xscl_data = wrf.to_np(scale_data["MAPFAC_MX"])
				yscl_data = wrf.to_np(scale_data["MAPFAC_MY"])
				scl_data = np.multiply(xscl_data,yscl_data)
			else:
				scl_data = []
		ens_data[ens_fcst,:,:] = wrf.to_np(inputvar)
	return ens_data, xsize_data, ysize_data, proj_data, xlim_data, ylim_data, dx_data, dy_data, wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_known_x, wprj_known_y, wprj_ref_lat, wprj_ref_lon, wprj_pole_lat, wprj_pole_lon, fcst_resp_box_mask, scl_data, lat_data, lon_data

# Read in ensemble data across multiple times, a wrapper around readensdata
def readensdata_mt (ens_dir, ens_mems, ens_times, ens_var, replacenan=0, preslevel=500, preslevel2=300, zagllevel=6, zagltop = 5000, zaglbottom = 2000, lat_bnds=(0,0), lon_bnds=(0,0), get_scl_data=False, def_fcst_resp_box=False, get_lat_lon=False):
	for ens_time in ens_times:
		if (ens_time == ens_times[0]):
			ens_data, xsize_data, ysize_data, proj_data, xlim_data, ylim_data, dx_data, dy_data, wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_known_x, wprj_known_y, wprj_ref_lat, wprj_ref_lon, wprj_pole_lat, wprj_pole_lon, fcst_resp_box_mask, scl_data, lat_data, lon_data = readensdata (ens_dir, ens_mems, ens_time, ens_var, replacenan, preslevel, preslevel2, zagllevel, zagltop, zaglbottom, lat_bnds, lon_bnds, get_scl_data, def_fcst_resp_box, get_lat_lon)
			ens_data_mt = np.zeros([len(ens_times),ens_data.shape[0],ens_data.shape[1],ens_data.shape[2]])
		else:
			ens_data = readensdata (ens_dir, ens_mems, ens_time, ens_var, replacenan, preslevel, preslevel2, zagllevel, zagltop, zaglbottom, lat_bnds, lon_bnds, get_scl_data, def_fcst_resp_box, get_lat_lon)[0]
		ens_data_mt[ens_time - ens_times[0]][:][:][:] = ens_data
	return ens_data_mt, xsize_data, ysize_data, proj_data, xlim_data, ylim_data, dx_data, dy_data, wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_known_x, wprj_known_y, wprj_ref_lat, wprj_ref_lon, wprj_pole_lat, wprj_pole_lon, fcst_resp_box_mask, scl_data, lat_data, lon_data

#dx and dy data are measured in meters
def calc_forecastresponse (ens_mems, fcst_resp_box_mask, fcst_data, fcst_resp_type = 1, fcst_resp_threshold = 0, scl_data = [], dx_data = 3000, dy_data = 3000):
	fcst_resp = np.zeros(ens_mems)
	for ens_fcst in np.arange(0,ens_mems,1):
		if (fcst_resp_type == 1):
			fcst_resp_mask = np.all([fcst_resp_box_mask,fcst_data[ens_fcst,:,:] >= fcst_resp_threshold],axis=0)
			fcst_resp[ens_fcst] = np.sum(scl_data[fcst_resp_mask])*dx_data*dy_data
		if (fcst_resp_type == 2):
			fcst_resp_mask = np.all([fcst_resp_box_mask,fcst_data[ens_fcst,:,:] <= fcst_resp_threshold],axis=0)
			fcst_resp[ens_fcst] = np.sum(scl_data[fcst_resp_mask])*dx_data*dy_data
		elif (fcst_resp_type == 3):
			fcst_data_2d = fcst_data[ens_fcst,:,:]
			fcst_resp[ens_fcst] = np.max(fcst_data_2d[fcst_resp_box_mask])
		elif (fcst_resp_type == 4):
			fcst_data_2d = fcst_data[ens_fcst,:,:]
			fcst_resp[ens_fcst] = np.min(fcst_data_2d[fcst_resp_box_mask])
		elif (fcst_resp_type == 5):
			fcst_data_2d = fcst_data[ens_fcst,:,:]
			fcst_resp[ens_fcst] = np.mean(fcst_data_2d[fcst_resp_box_mask])
	return fcst_resp

#Calculate forecast response across times
def calc_forecastresponse_mt (ens_mems, fcst_resp_box_mask, fcst_data_mt, fcst_resp_type = 1, fcst_resp_threshold = 0, time_operation_type = 0, scl_data = [], dx_data = 3000, dy_data = 3000):
	fcst_resp_mt = np.zeros([fcst_data_mt.shape[0],ens_mems])
	for fcst_time in np.arange(0,fcst_data_mt.shape[0],1):
		fcst_resp_st = calc_forecastresponse (ens_mems, fcst_resp_box_mask, fcst_data_mt[fcst_time][:][:][:], fcst_resp_type, fcst_resp_threshold, scl_data, dx_data, dy_data)
		fcst_resp_mt[fcst_time][:] = fcst_resp_st
	if (time_operation_type == 1):
		fcst_resp = np.max(fcst_resp_mt, axis=0)
	elif (time_operation_type == 2):
		fcst_resp = np.min(fcst_resp_mt, axis=0)
	else:
		fcst_resp = np.mean(fcst_resp_mt, axis=0)
	return fcst_resp

#Aggregate gridded data along the time dimension when multiple times are read in
def time_aggregate_ens_grids (ens_tdata, time_operation_type):
	if (time_operation_type == 1):
		ens_data = np.max(ens_tdata, axis=0)
	elif (time_operation_type == 2):
		ens_data = np.min(ens_tdata, axis=0)
	else:
		ens_data = np.mean(ens_tdata, axis=0)
	return ens_data

def calc_esa (pert_data, fcst_resp):
	pval_data = np.zeros((pert_data.shape[1], pert_data.shape[2]))
	rval_data = np.zeros((pert_data.shape[1], pert_data.shape[2]))
	sval_data = np.zeros((pert_data.shape[1], pert_data.shape[2]))
	for i in np.arange(0,pert_data.shape[1],1):
		for j in np.arange(0,pert_data.shape[2],1):
			ens_perts = pert_data[:,i,j]
			rval, pval = stats.pearsonr(ens_perts,fcst_resp)
			sval = np.std(ens_perts)
			rval_data[i,j] = rval
			pval_data[i,j] = pval
			sval_data[i,j] = sval
	return rval_data,pval_data,sval_data

# Attempting to distinguish spurious correlations from real signals by calculating a heuristic.
def calc_heuristic (pval_data, pval_threshold, radius):
	pval_test = pval_data <= pval_threshold
	kernel = skimage.morphology.disk(radius)
	heuristic1 = signal.convolve2d(pval_test.astype(int), kernel, mode = 'same')
	heuristic2 = signal.convolve2d(np.ones(pval_test.shape), kernel, mode = 'same')
	heuristic = heuristic1/heuristic2
	return heuristic

# Apply heuristic and p-value filters and create a mask.
#
# If we our p-value threshold is 0.05, for example, if 5% of our data appear to be statistically significant,
# it is likely that it is due to random chance.  If the signal is much greater than our threshold, e.g., 40% of
# points in a region satisfy the p-value threshold, it suggests that there may actually be a relationship in
# the data and that it's not just statistical noise.  There's a second threshold here, where we need a certain
# proportion of points in a region to satisfy the significance test, otherwise we treat it as random noise.
# We implement that as a multiplier of our p-value threshold.  For example, if our p-value threshold is 0.05
# and we want 30% (0.3) of points in a region to satisfy the threshold before we consider them, we will set our
# multiplier to 6.  That means we need 6 times as many points to satisfy the threshold as would be expected by
# random chance in order to pass the filter.
#
# To skip the heuristic filter, set the multiplier to zero and/or set the heuristic array to [].
#
# This function applies both the aforementioned heuristic and the p-value filters.
def apply_pval_heuristic_filters (pval_data, pval_threshold, heuristic_data, heuristic_multiplier):
	filterarea = pval_data <= pval_threshold
	if (heuristic_data != []):
		heuristic_threshold = heuristic_multiplier * pval_threshold
		heuristicarea = heuristic_data >= heuristic_threshold
		filterarea = np.all([filterarea, heuristicarea],axis=0)
	return filterarea

# Plot data on a map
def plot_map_simple (plotdata, figxsize, figysize, shp1, shp2, colorlevels, xlim_data, ylim_data, lat_data, lon_data, proj_data, outputfile, figuretitle = '', shp1color = 'black', shp2color = 'black', shp1width = 0.25, shp2width = 0.5, coastlinewidth = 0.8, coastlinecolor = 'black', boxcolor = 'black', gridlinecolor = 'black', gridlinestyle = 'dotted', colorpalette = 'RdBu_r', lat_bnds = [], lon_bnds = [], bounded_region = [], boundedwidth = 1):
	fig = plt.figure(figsize=(figxsize,figysize))
	ax = plt.axes(projection=proj_data)
	if (shp1 != ""):
		shpfeature1 = feature.ShapelyFeature(shapereader.Reader(shp1).geometries(),crs.PlateCarree(), edgecolor=shp1color, facecolor='none')
		ax.add_feature(shpfeature1, linewidth=shp1width, edgecolor=shp1color)
	if (shp2 != ""):
		shpfeature2 = feature.ShapelyFeature(shapereader.Reader(shp2).geometries(),crs.PlateCarree(), edgecolor=shp2color, facecolor='none')
		ax.add_feature(shpfeature2, linewidth=shp2width, edgecolor=shp2color)
	ax.coastlines('50m', linewidth=coastlinewidth, edgecolor=coastlinecolor)
	plt.contourf(lon_data, lat_data, plotdata, levels=colorlevels, transform=crs.PlateCarree(), cmap=cm.get_cmap(colorpalette),zorder=0)
	if (bounded_region != []):
		plt.contour(lon_data, lat_data, bounded_region, colors=[boxcolor], linewidths=[boundedwidth], levels=[0.5], transform=crs.PlateCarree(),zorder=4)
	elif ((len(lat_bnds) == 2) and (len(lon_bnds) == 2)):
		plt.plot([lon_bnds[0],lon_bnds[0]],[lat_bnds[0],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[1],lon_bnds[1]],[lat_bnds[0],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[0],lon_bnds[1]],[lat_bnds[0],lat_bnds[0]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[0],lon_bnds[1]],[lat_bnds[1],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
	if (len(figuretitle) > 0):
		plt.title(figuretitle)
	plt.colorbar(ax=ax, shrink=.98)
	ax.set_xlim(xlim_data)
	ax.set_ylim(ylim_data)
	ax.gridlines(color=gridlinecolor, linestyle=gridlinestyle)
	plt.savefig(outputfile)
	plt.close(fig)

# Plot data on a map, outlining a region where the boolean value is True
def plot_map_outlinebool (plotdata, booldata, figxsize, figysize, shp1, shp2, colorlevels, xlim_data, ylim_data, lat_data, lon_data, proj_data, outputfile, figuretitle = '', shp1color = 'black', shp2color = 'black', shp1width = 0.25, shp2width = 0.5, coastlinewidth = 0.8, coastlinecolor = 'black', boollinewidth = 1.0, boollinecolor = 'black', boxcolor = 'black', gridlinecolor = 'black', gridlinestyle = 'dotted', colorpalette = 'RdBu_r', lat_bnds = [], lon_bnds = [], bounded_region = [], boundedwidth = 1):
	plot_map_contouroverlay (plotdata, booldata.astype(int), figxsize, figysize, shp1, shp2, colorlevels, [0.5], xlim_data, ylim_data, lat_data, lon_data, proj_data, outputfile, figuretitle = figuretitle, shp1color = shp1color, shp2color = shp2color, shp1width = shp1width, shp2width = shp2width, coastlinewidth = coastlinewidth, coastlinecolor = coastlinecolor, overlaylinewidth = boollinewidth, overlaylinecolor = boollinecolor, boxcolor = boxcolor, gridlinecolor = gridlinecolor, gridlinestyle = gridlinestyle, colorpalette = colorpalette, lat_bnds = lat_bnds, lon_bnds = lon_bnds, bounded_region = bounded_region, boundedwidth = boundedwidth)

# Plot data on a map, outlining a region where the boolean value is True
def plot_map_contouroverlay (plotdata, overlaydata, figxsize, figysize, shp1, shp2, colorlevels, overlaylevels, xlim_data, ylim_data, lat_data, lon_data, proj_data, outputfile, figuretitle = '', shp1color = 'black', shp2color = 'black', shp1width = 0.25, shp2width = 0.5, coastlinewidth = 0.8, coastlinecolor = 'black', overlaylinewidth = 1.0, overlaylinecolor = 'black', boxcolor = 'black', gridlinecolor = 'black', gridlinestyle = 'dotted', colorpalette = 'RdBu_r', lat_bnds = [], lon_bnds = [], bounded_region = [], boundedwidth = 1):
	fig = plt.figure(figsize=(figxsize,figysize))
	ax = plt.axes(projection=proj_data)
	if (shp1 != ""):
		shpfeature1 = feature.ShapelyFeature(shapereader.Reader(shp1).geometries(),crs.PlateCarree(), edgecolor=shp1color, facecolor='none')
		ax.add_feature(shpfeature1, linewidth=shp1width, edgecolor=shp1color)
	if (shp2 != ""):
		shpfeature2 = feature.ShapelyFeature(shapereader.Reader(shp2).geometries(),crs.PlateCarree(), edgecolor=shp2color, facecolor='none')
		ax.add_feature(shpfeature2, linewidth=shp2width, edgecolor=shp2color)
	ax.coastlines('50m', linewidth=coastlinewidth, edgecolor=coastlinecolor)
	plt.contourf(lon_data, lat_data, plotdata, levels=colorlevels, transform=crs.PlateCarree(), cmap=cm.get_cmap(colorpalette),zorder=0)
	plt.colorbar(ax=ax, shrink=.98)
	plt.contour(lon_data, lat_data, overlaydata, colors=[overlaylinecolor], linewidths=[overlaylinewidth], levels=overlaylevels, transform=crs.PlateCarree(),zorder=4)
	if (bounded_region != []):
		plt.contour(lon_data, lat_data, bounded_region, colors=[boxcolor], linewidths=[boundedwidth], levels=[0.5], transform=crs.PlateCarree(),zorder=4)
	elif ((len(lat_bnds) == 2) and (len(lon_bnds) == 2)):
		plt.plot([lon_bnds[0],lon_bnds[0]],[lat_bnds[0],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[1],lon_bnds[1]],[lat_bnds[0],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[0],lon_bnds[1]],[lat_bnds[0],lat_bnds[0]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[0],lon_bnds[1]],[lat_bnds[1],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
	if (len(figuretitle) > 0):
		plt.title(figuretitle)
	ax.set_xlim(xlim_data)
	ax.set_ylim(ylim_data)
	ax.gridlines(color=gridlinecolor, linestyle=gridlinestyle)
	plt.savefig(outputfile)
	plt.close(fig)

# Plot data and obs with a color scale on a map
def plot_map_col_obs (plotdata, figxsize, figysize, shp1, shp2, colorlevels, xlim_data, ylim_data, lat_data, lon_data, proj_data, obs_vals, obs_lats, obs_lons, obs_vmin, obs_vmax, outputfile, figuretitle = '', shp1color = 'black', shp2color = 'black', shp1width = 0.25, shp2width = 0.5, coastlinewidth = 0.8, coastlinecolor = 'black', boxcolor = 'black', gridlinecolor = 'black', gridlinestyle = 'dotted', colorpalette = 'jet', obs_colorpalette = 'RdBu_r', lat_bnds = [], lon_bnds = [], bounded_region = [], boundedwidth = 1):
	fig = plt.figure(figsize=(figxsize,figysize))
	ax = plt.axes(projection=proj_data)
	if (shp1 != ""):
		shpfeature1 = feature.ShapelyFeature(shapereader.Reader(shp1).geometries(),crs.PlateCarree(), edgecolor=shp1color, facecolor='none')
		ax.add_feature(shpfeature1, linewidth=shp1width, edgecolor=shp1color)
	if (shp2 != ""):
		shpfeature2 = feature.ShapelyFeature(shapereader.Reader(shp2).geometries(),crs.PlateCarree(), edgecolor=shp2color, facecolor='none')
		ax.add_feature(shpfeature2, linewidth=shp2width, edgecolor=shp2color)
	ax.coastlines('50m', linewidth=coastlinewidth, edgecolor=coastlinecolor)
	plt.contourf(lon_data, lat_data, plotdata, levels=colorlevels, transform=crs.PlateCarree(), cmap=cm.get_cmap(colorpalette),zorder=0)
	plt.colorbar(ax=ax, shrink=.98)
	ax.scatter(obs_lons, obs_lats, c=obs_vals, s=40, edgecolor='black', linewidth=0.5, transform=crs.PlateCarree(), cmap=cm.get_cmap(obs_colorpalette), vmin=obs_vmin, vmax=obs_vmax, zorder=3)
	if (bounded_region != []):
		plt.contour(lon_data, lat_data, bounded_region, colors=[boxcolor], linewidths=[boundedwidth], levels=[0.5], transform=crs.PlateCarree(),zorder=4)
	elif ((len(lat_bnds) == 2) and (len(lon_bnds) == 2)):
		plt.plot([lon_bnds[0],lon_bnds[0]],[lat_bnds[0],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[1],lon_bnds[1]],[lat_bnds[0],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[0],lon_bnds[1]],[lat_bnds[0],lat_bnds[0]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
		plt.plot([lon_bnds[0],lon_bnds[1]],[lat_bnds[1],lat_bnds[1]],color=boxcolor, transform=crs.PlateCarree(),zorder=4)
	if (len(figuretitle) > 0):
		plt.title(figuretitle)
	ax.set_xlim(xlim_data)
	ax.set_ylim(ylim_data)
	ax.gridlines(color=gridlinecolor, linestyle=gridlinestyle)
	plt.savefig(outputfile)
	plt.close(fig)
	
# Read in the MADIS observations from a file
def read_madis_obs (madis_file, param_name = 'temperature', lat_name = 'latitude', lon_name = 'longitude', qc_name = 'temperatureQCR'):
	ncfile = ncdf.Dataset(madis_file)
	madis_lats = ncfile.variables[lat_name][:]
	madis_lons = ncfile.variables[lon_name][:]
	madis_obs = ncfile.variables[param_name][:]
	if (qc_name != ''):
		madis_qc = ncfile.variables[qc_name][:]
	else:
		madis_qc = []
	return madis_obs, madis_lats, madis_lons, madis_qc

# Read in a specific MADIS variable, which can be a derived product.  This returns the variable and QC data for the variable.
def read_madis_var (madis_file, param_name):
	ncfile = ncdf.Dataset(madis_file)
	if (param_name == 't2'):
		madis_obs = ncfile.variables['temperature'][:]
		madis_qc = ncfile.variables['temperatureQCR'][:]
	elif (param_name == 'td2'):
		madis_obs = ncfile.variables['dewpoint'][:]
		madis_qc = ncfile.variables['dewpointQCR'][:]
	elif (param_name == 'slp'):
		madis_obs = ncfile.variables['seaLevelPress'][:]
		madis_qc = ncfile.variables['seaLevelPressQCR'][:]
	elif (param_name == 'u10'):
		wspd_obs = ncfile.variables['windSpeed'][:]
		wspd_qc = ncfile.variables['windSpeedQCR'][:]
		wdir_obs = ncfile.variables['windDir'][:]
		wdir_qc = ncfile.variables['windDirQCR'][:]
		madis_obs = np.multiply(np.multiply(np.sin(np.multiply(wdir_obs,np.pi/180)),-1),wspd_obs)
		madis_qc = np.bitwise_or(wspd_qc,wdir_qc)
	elif (param_name == 'v10'):
		wspd_obs = ncfile.variables['windSpeed'][:]
		wspd_qc = ncfile.variables['windSpeedQCR'][:]
		wdir_obs = ncfile.variables['windDir'][:]
		wdir_qc = ncfile.variables['windDirQCR'][:]
		madis_obs = np.multiply(np.multiply(np.cos(np.multiply(wdir_obs,np.pi/180)),-1),wspd_obs)
		madis_qc = np.bitwise_or(wspd_qc,wdir_qc)
	elif (param_name == 'lat'):
		madis_obs = ncfile.variables['latitude'][:]
		madis_qc = []
	elif (param_name == 'lon'):
		madis_obs = ncfile.variables['longitude'][:]
		madis_qc = []
	else:
		madis_obs = []
		madis_qc = []
	return madis_obs, madis_qc

# Filter the MADIS observations to the grid, and convert the latitude and longitude values to (x,y) pairs
#
# We remove observations near the edge by setting edge_filter_width to the number of grid points near the
# edge for which we want to discard observations.  By default, this is set to zero grid points, but
# probably should be larger.
def madis_filter_to_grid_xy (madis_obs, madis_lats, madis_lons, xsize_data, ysize_data, wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_ref_lat, wprj_ref_lon, wprj_known_x, wprj_known_y, wprj_pole_lat, wprj_pole_lon, dx_data, dy_data, edge_filter_width = 0, madis_qc = []):
	wrf_madis_xy_unfiltered = wrf.ll_to_xy_proj(madis_lats,madis_lons,meta=False,map_proj=wprj_map_proj,truelat1=wprj_truelat1,truelat2=wprj_truelat2,stand_lon=wprj_stand_lon,ref_lat=wprj_ref_lat,ref_lon=wprj_ref_lon,known_x=wprj_known_x,known_y=wprj_known_y,pole_lat=wprj_pole_lat,pole_lon=wprj_pole_lon,dx=dx_data,dy=dy_data)
	wrf_madis_x_unfiltered = wrf_madis_xy_unfiltered[0]
	wrf_madis_y_unfiltered = wrf_madis_xy_unfiltered[1]
	wrf_madis_mask = np.all([(wrf_madis_x_unfiltered >= edge_filter_width),(wrf_madis_x_unfiltered < (xsize_data - edge_filter_width)),(wrf_madis_y_unfiltered >= edge_filter_width),(wrf_madis_y_unfiltered < (ysize_data - edge_filter_width))],axis=0)
	wrf_madis_x = wrf_madis_x_unfiltered[wrf_madis_mask]
	wrf_madis_y = wrf_madis_y_unfiltered[wrf_madis_mask]
	wrf_madis_lats = madis_lats[wrf_madis_mask]
	wrf_madis_lons = madis_lons[wrf_madis_mask]
	wrf_madis_obs = madis_obs[wrf_madis_mask]
	if (madis_qc != []):
		wrf_madis_qc = madis_obs[wrf_madis_mask]
	else:
		wrf_madis_qc = []
	return wrf_madis_obs, wrf_madis_x, wrf_madis_y, wrf_madis_lats, wrf_madis_lons, wrf_madis_qc

# Filter the observations to only use those with no NaN values and those in areas of significant sensitivity
# We also filter to see if the QC value is 0, and discard locations where the QC value isn't zero
def filter_madis_obs (wrf_madis_obs, wrf_madis_x, wrf_madis_y, pval_mask, wrf_madis_lats = [], wrf_madis_lons = [], wrf_madis_qc = [], center_x = 0, center_y = 0, filter_radius = 0):
	# Note: WRF arrays are (y,x)
	pert_pval_mask = pval_mask[wrf_madis_y,wrf_madis_x]
	if (filter_radius > 0):
		radius_mask = np.zeros(pval_mask.shape)
		mxmax = radius_mask.shape[1] - 1
		mymax = radius_mask.shape[0] - 1
		filter_shape = np.array(skimage.morphology.disk(filter_radius))
		if ((center_x - filter_radius) < 0):
			mx1 = 0
			fx1 = filter_radius - center_x
		else:
			mx1 = center_x - filter_radius
			fx1 = 0
		if ((center_y - filter_radius) < 0):
			my1 = 0
			fy1 = filter_radius - center_y
		else:
			my1 = center_y - filter_radius
			fy1 = 0
		if ((center_x + filter_radius) > mxmax):
			mx2 = mxmax
			fx2 = (filter_radius * 2) - ((center_x + filter_radius) - mxmax)
		else:
			mx2 = center_x + filter_radius
			fx2 = filter_radius * 2
		if ((center_y + filter_radius) > mymax):
			my2 = mxmax
			fy2 = (filter_radius * 2) - ((center_y + filter_radius) - mxmax)
		else:
			my2 = center_y + filter_radius
			fy2 = filter_radius * 2
		radius_mask[my1:my2,mx1:mx2] = filter_shape[fy1:fy2,fx1:fx2]
		radius_pval_mask = radius_mask.astype(bool)[wrf_madis_y,wrf_madis_x]
		pert_pval_mask = np.all([pert_pval_mask,radius_pval_mask],axis=0)
	pert_unfiltered_notmissing_mask = ~(wrf_madis_obs.mask)
	if (wrf_madis_qc == []):
		pert_err_mask = np.all([pert_pval_mask,pert_unfiltered_notmissing_mask],axis=0)
	else:
		qc_mask = (wrf_madis_qc == 0)
		pert_err_mask = np.all([pert_pval_mask,pert_unfiltered_notmissing_mask,qc_mask],axis=0)
	f_wrf_madis_x = wrf_madis_x[pert_err_mask]
	f_wrf_madis_y = wrf_madis_y[pert_err_mask]
	f_wrf_madis_obs = wrf_madis_obs[pert_err_mask]
	if (wrf_madis_lats != []):
		f_wrf_madis_lats = wrf_madis_lats[pert_err_mask]
	else:
		f_wrf_madis_lats = []
	if (wrf_madis_lons != []):
		f_wrf_madis_lons = wrf_madis_lons[pert_err_mask]
	else:
		f_wrf_madis_lons = []
	return f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, f_wrf_madis_lats, f_wrf_madis_lons

# Calculate the error of each ensemble member in sensitivity/heuristic-masked areas
# method = 0 uses the rval squared to weight each point
# method = 1 is the same as method 0, except without squaring the error of each observation
# method = 2 accounts for the standard deviation of perturbations, and squares the error of each observation like method = 0
# method = 3 accounts for the standard deviation of perturbations, without squaring the error of each observations
#
# If weight_radius is greater than zero, it is a distance in grid points.  Observations are weighted based on
# the number of observations within that radius of the grid point.  For example, for an observation, if there
# are no other observations within the radius, it is weighted 1.  If there are three other observations within
# the radius, it is weighted 0.25.  This is an attempt to account for the uneven spatial distribution of
# observations.
def calc_ens_member_err (f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, pert_data, rval_data, sval_data = [], method = 0, weight_radius = 0):
	ens_mems = pert_data.shape[0]
	ens_mem_err = np.zeros(ens_mems)
	if (weight_radius > 0):
		ob_count_grid = np.zeros(rval_data.shape)
		for i in np.arange(0,len(f_wrf_madis_obs),1):
			ob_count_grid[f_wrf_madis_y[i],f_wrf_madis_x[i]] = ob_count_grid[f_wrf_madis_y[i],f_wrf_madis_x[i]] + 1
		kernel = skimage.morphology.disk(weight_radius)
		obs_count_neighborhood = signal.convolve2d(ob_count_grid, kernel, mode = 'same')
		ob_weight = np.divide(1,obs_count_neighborhood[f_wrf_madis_y,f_wrf_madis_x])
	else:
		ob_weight = np.ones(f_wrf_madis_obs.shape)
	if (len(f_wrf_madis_obs) > 0):
		for ens_mem_num in np.arange(0,ens_mems,1):
			ens_pert = pert_data[ens_mem_num,:,:]
			# Note: WRF arrays are (y,x)
			pert_sim_obs = ens_pert[f_wrf_madis_y,f_wrf_madis_x]
			pert_err_unw = f_wrf_madis_obs - pert_sim_obs
			pert_err = np.multiply(pert_err_unw,ob_weight)
			if (method == 1):
				# Note: WRF arrays are (y,x)
				pert_sim_obspert_err = np.absolute(np.multiply(pert_err,rval_data[f_wrf_madis_y,f_wrf_madis_x]))
				ens_mem_err[ens_mem_num] = np.sum(pert_sim_obspert_err)
			elif (method == 2):
				# Note: WRF arrays are (y,x)
				pert_sim_obspert_err = np.absolute(np.multiply(np.divide(pert_err,sval_data[f_wrf_madis_y,f_wrf_madis_x]),rval_data[f_wrf_madis_y,f_wrf_madis_x]))
				ens_mem_err[ens_mem_num] = np.sum(np.square(pert_sim_obspert_err))
			elif (method == 3):
				# Note: WRF arrays are (y,x)
				pert_sim_obspert_err = np.absolute(np.multiply(np.divide(pert_err,sval_data[f_wrf_madis_y,f_wrf_madis_x]),rval_data[f_wrf_madis_y,f_wrf_madis_x]))
				ens_mem_err[ens_mem_num] = np.sum(pert_sim_obspert_err)
			else:
				# Note: WRF arrays are (y,x)
				pert_sim_obspert_err = np.absolute(np.multiply(pert_err,rval_data[f_wrf_madis_y,f_wrf_madis_x]))
				ens_mem_err[ens_mem_num] = np.sum(np.square(pert_sim_obspert_err))
	return ens_mem_err

# Calculate the ensemble mean
def calc_ens_mean (var_data):
	ens_mean = np.mean(var_data,axis=0)
	return ens_mean

# Calculate perturbations and, optionally, standardized perturbations between observations
# and a gridded variable
def calc_obs_pert (obs_data, obs_x, obs_y, grid_data, r_grid_data = [], s_grid_data = []):
	# Note: WRF arrays are (y,x)
	grid_obs = grid_data[obs_y, obs_x]
	obs_diff = np.subtract(obs_data, grid_obs)
	if (r_grid_data != []):
		# Note: WRF arrays are (y,x)
		r_grid_obs = r_grid_data[obs_y, obs_x]
		r_obs_diff = np.multiply(obs_diff, r_grid_obs)
	else:
		r_obs_diff = []
		rs_obs_diff = []
	if (s_grid_data != []):
		# Note: WRF arrays are (y,x)
		s_grid_obs = s_grid_data[obs_y, obs_x]
		s_obs_diff = np.divide(obs_diff, s_grid_obs)
		if (r_grid_data != []):
			rs_obs_diff = np.multiply(s_obs_diff, r_grid_obs)
	else:
		s_obs_diff = []
		rs_obs_diff = []
	return obs_diff, r_obs_diff, s_obs_diff, rs_obs_diff

# Convert Kelvins to Fahrenheit
def conv_temp_k_to_f (data_array):
	new_data_array = ((data_array - 273.15) * (9.0/5.0)) + 32.0
	return new_data_array

# Convert Kelvins to Celsius
def conv_temp_k_to_c (data_array):
	new_data_array = data_array - 273.15
	return new_data_array

# Convert Kelvins to Celsius
def conv_temp_c_to_k (data_array):
	new_data_array = data_array + 273.15
	return new_data_array

# Iterate through each ensemble member and run a local maximum filter, creating a new image
# where each pixel is the result of the filter run within a set radius in the original image.
# This can be a median filter (filter_type = 0), maximum filter (filter_type = 1), or a
# minimum filter (filter_type = 2).
def ens_neighborhood_filter (ens_data, filter_type, filter_radius):
	filtered_data = np.zeros(ens_data.shape)
	if (len(ens_data.shape) == 2):
		img_input = ens_data
		if (filter_type == 1):
			filtered_data = ndi.maximum_filter(img_input, size=filter_radius)
		elif (filter_type == 2):
			filtered_data = ndi.minimum_filter(img_input, size=filter_radius)
		else:
			filtered_data = ndi.median_filter(img_input, size=filter_radius)
	else:
		for ens_mem in np.arange(0,ens_data.shape[0],1):
			img_input = ens_data[ens_mem][:][:]
			if (filter_type == 1):
				filtered_data[ens_mem][:][:] = ndi.maximum_filter(img_input, size=filter_radius)
			elif (filter_type == 2):
				filtered_data[ens_mem][:][:] = ndi.minimum_filter(img_input, size=filter_radius)
			else:
				filtered_data[ens_mem][:][:] = ndi.median_filter(img_input, size=filter_radius)
	return filtered_data

# Calculate the probability of exceedance (for prob_mode, set 0 for value >=, 1 for value <=)
def calc_unw_prob_exceedance (ens_data, threshold, prob_mode):
	if (prob_mode == 1):
		masked_ens = (ens_data <= threshold)
	else:
		masked_ens = (ens_data >= threshold)
	prob_exceed = np.sum(masked_ens,axis = 0) / (ens_data.shape[0])
	return prob_exceed

# Calculate the ensemble standard deviation
def calc_ens_sd (ens_data):
	ens_sd = np.std(ens_data,axis=0)
	return ens_sd

# Calculate the proportion of points within a threshold inside a radius.  We use this to identify
# regions of uncertainty, which may be good candidates for the forecast response region.
def calc_pts_in_threshold (grid2d, filter_radius, min_threshold, max_threshold):
	threshold_test = np.all([(grid2d >= min_threshold), (grid2d <= max_threshold)],axis = 0)
	kernel = skimage.morphology.disk(filter_radius)
	test1 = signal.convolve2d(threshold_test.astype(int), kernel, mode = 'same')
	test2 = signal.convolve2d(np.ones(threshold_test.shape), kernel, mode = 'same')
	proportion = test1/test2
	return proportion

# Calculate the proportion of points exceeding a two thresholds within a radius.  For example, we may
# want to identify points where the probability of a thunderstorm is high (e.g., >= 95%) but there's
# large variability in other parameters like updraft helicity.  We also have two testing mode
# variables, one for each of the variables we are thresholding.  The variables are set accordingly:
# for prob_mode, set 0 for value >=, 1 for value <=.  We also calculate probabilities within a
# specified neighborhood.
def calc_prob_dual_exceedance (ens_data1, threshold1, test_mode1, ens_data2, threshold2, test_mode2, filter_radius):
	if (test_mode1 == 1):
		masked_ens1 = (ens_data1 <= threshold1)
	else:
		masked_ens1 = (ens_data1 >= threshold1)
	if (test_mode2 == 1):
		masked_ens2 = (ens_data2 <= threshold2)
	else:
		masked_ens2 = (ens_data2 >= threshold2)
	masked_ens = np.all([masked_ens1, masked_ens2],axis=0)
	kernel = skimage.morphology.disk(filter_radius)
	test1 = signal.convolve2d(masked_ens.astype(int), kernel, mode = 'same')
	test2 = signal.convolve2d(np.ones(masked_ens.shape), kernel, mode = 'same')
	proportion = test1/test2
	return proportion

# Label features in an image that are connected.  The features are identified based on coherent regions
# that exceed a threshold.  For threshold_mode, set 0 for value >=, 1 for value <=.
def label_regions (data_array, threshold, threshold_mode, close_iterations = 0, dilate_iterations = 0, edge_filter = 0, min_size = 0):
	if (threshold_mode == 1):
		data_mask = (data_array <= threshold)
	else:
		data_mask = (data_array >= threshold)
	if (close_iterations > 0):
		data_mask = ndi.binary_closing(data_mask, iterations = close_iterations)
	if (dilate_iterations > 0):
		data_mask = ndi.binary_dilation(data_mask, iterations = dilate_iterations)
	regionmap, regionmap_max = ndi.label(data_mask.astype(int), structure = ndi.generate_binary_structure(2,2))
	if ((edge_filter > 0) or (min_size > 0)):
		old_data_mask = data_mask
	if (edge_filter > 0):
		for cur_region in np.arange(1,regionmap_max+1,1):
			center_loc = ndi.center_of_mass(old_data_mask, regionmap, cur_region)
			if ((center_loc[0] < edge_filter) or (center_loc[0] >= (old_data_mask.shape[0] - edge_filter)) or (center_loc[1] < edge_filter) or (center_loc[1] >= (old_data_mask.shape[1] - edge_filter))):
				region_mask = (regionmap == cur_region)
				data_mask[region_mask] = 0
	if (min_size > 0):
		for cur_region in np.arange(1,regionmap_max+1,1):
			region_size = np.sum(regionmap == cur_region)
			if (region_size < min_size):
				region_mask = (regionmap == cur_region)
				data_mask[region_mask] = 0
	if ((edge_filter > 0) or (min_size > 0)):
		regionmap, regionmap_max = ndi.label(data_mask.astype(int), structure = ndi.generate_binary_structure(2,2))
	return regionmap, regionmap_max

# Get the center of a region
def get_region_center (regionmap, regionid):
	center_y,center_x = ndi.center_of_mass(regionmap==regionid)
	return int(round(center_x,0)), int(round(center_y,0))

# Draw a basic scatter plot (e.g., member error vs. forecast response) and save to a file
def plot_basic_scatter (xvar, yvar, figxsize, figysize, outputfile, figuretitle="", xlabel = "", ylabel = ""):
	fig = plt.figure(figsize=(figxsize,figysize))
	ax = plt.axes()
	ax.scatter(xvar,yvar)
	if (len(figuretitle) > 0):
		plt.title(figuretitle)
	if (len(xlabel) > 0):
		ax.set_xlabel(xlabel)
	if (len(ylabel) > 0):
		ax.set_ylabel(ylabel)
	plt.savefig(outputfile)
	plt.close(fig)

# Plot a stacked bar chart with data
def plot_stacked_bar_chart (var_data, figxsize, figysize, outputfile, figuretitle="", x_labels = [], data_labels = [], bar_width = 0.5, ylabel = "", legend_loc = "lower right"):
	if (x_labels == []):
		bar_labels = []
		for i in np.arange(0,var_data.shape[1],1):
			bar_labels.append(str(i+1))
	else:
		bar_labels = x_labels
	fig = plt.figure(figsize=(figxsize,figysize))
	ax = plt.axes()
	bottom_level = np.zeros(var_data.shape[1])
	for i in np.arange(0,var_data.shape[0],1):
		if (data_labels == []):
			ax.bar(bar_labels, var_data[i][:], bar_width, bottom=bottom_level)
		else:
			ax.bar(bar_labels, var_data[i][:], bar_width, bottom=bottom_level, label=data_labels[i])
		bottom_level = np.add(bottom_level,var_data[i][:])
	if (data_labels != []):
		ax.legend(loc = legend_loc)
	if (len(figuretitle) > 0):
		plt.title(figuretitle)
	if (len(ylabel) > 0):
		ax.set_ylabel(ylabel)
	plt.savefig(outputfile)
	plt.close(fig)
