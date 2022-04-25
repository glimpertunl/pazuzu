from __future__ import print_function
import os
import operator
import numpy as np
import time
import pazuzu

madis_file = '/path/to/your/madis_data/20200429_0100.nc'
data_dir = '/path/to/your/wofs_cases/'
case_dir = '20200428'
case_time = '0000'
first_time = 1
ens_mems = 36
var0_time = [12]
var0_varlist = ['t2','td2','slp','u10','v10']
var0_labels = ['Temperature','Dewpoint','Sea Level Pressure','U Wind Component','V Wind Component']
var0_zaglbottom = 0
var0_zagltop = 0
var1_time = np.arange(24,25,1)
var2_time = np.arange(21,28,1)
var1_threshold = 40
var2_threshold = 200
var1_var = 'mdbz'
var2_var = 'uh_zagl'
var1_zaglbottom = 0
var1_zagltop = 0
var2_zaglbottom = 2000
var2_zagltop = 5000
neighborhood_filter_radius = 10
pval_threshold = 0.1
heuristic_multiplier = 2.5
heuristic_radius = 30
madis_filter_radius = 75
obs_weight_radius = 20
countyshp = '/path/to/your/shapefiles/cb_2018_us_county_5m.shp'
stateshp = '/path/to/your/shapefiles/cb_2018_us_state_5m.shp'
output_dir = '/path/to/your/output/'

os.environ['TZ']='UTC'

ens_dir = data_dir + '/' + case_dir + '/WRFOUT/' + case_time

var1_tdata, xsize_data, ysize_data, proj_data, xlim_data, ylim_data, dx_data, dy_data,wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_known_x, wprj_known_y, wprj_ref_lat, wprj_ref_lon, wprj_pole_lat, wprj_pole_lon, scl_data, lat_data, lon_data = operator.itemgetter(*range(0,18),19,20,21)(pazuzu.readensdata_mt(ens_dir, ens_mems, var1_time, var1_var, zaglbottom=var1_zaglbottom, zagltop=var1_zagltop, get_lat_lon=True, get_scl_data=True))
var1_data = pazuzu.time_aggregate_ens_grids (var1_tdata, 0)
var1_filtered = pazuzu.ens_neighborhood_filter(var1_data, 1, neighborhood_filter_radius)
var1_exceedance = pazuzu.calc_unw_prob_exceedance (var1_filtered, var1_threshold, 0)
var1_sd = pazuzu.calc_ens_sd(var1_filtered)
var1_uncertainty = pazuzu.calc_pts_in_threshold (var1_exceedance, neighborhood_filter_radius, 0.25, 0.75)
var1_mean = pazuzu.calc_ens_mean(var1_data)
var1_regions, var1_regions_max = pazuzu.label_regions(var1_uncertainty, 0.5, 0, close_iterations = 3, dilate_iterations = 2, edge_filter = 10)
pazuzu.plot_map_outlinebool (var1_exceedance, var1_mean >= 40, 8, 6, countyshp, stateshp, np.arange(-1.00,1.05,0.05), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/exceedance_overlay_'+var1_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var1_exceedance, 8, 6, countyshp, stateshp, np.arange(0,1.025,0.025), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/exceedance_'+var1_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var1_uncertainty, 8, 6, countyshp, stateshp, np.arange(0,1.025,0.025), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/uncertainty_'+var1_var+'_'+case_dir+'_'+case_time+'.png')	
pazuzu.plot_map_simple (var1_sd, 8, 6, countyshp, stateshp, np.arange(0,30.5,0.5), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/sd_'+var1_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var1_regions, 8, 6, countyshp, stateshp, np.arange(-0.5,var1_regions_max+1.5,1), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/regions_'+var1_var+'_'+case_dir+'_'+case_time+'.png')

for cur_region in np.arange(1,var1_regions_max+1,1):
	center_x, center_y = pazuzu.get_region_center(var1_regions, cur_region)
	fcst_resp_mask = (var1_regions == cur_region)
	fcst_resp = pazuzu.calc_forecastresponse_mt (ens_mems, fcst_resp_mask, var1_tdata, fcst_resp_type = 1, fcst_resp_threshold=var1_threshold, time_operation_type = 1, scl_data=scl_data,dx_data=dx_data,dy_data=dy_data)
	ens_mem_err = np.zeros([len(var0_varlist),ens_mems])
	for i in np.arange(0,len(var0_varlist),1):
		var0_var = var0_varlist[i]
		var0_tdata = pazuzu.readensdata_mt(ens_dir, ens_mems, var0_time, var0_var, zaglbottom=var0_zaglbottom, zagltop=var0_zagltop)[0]
		var0_data = pazuzu.time_aggregate_ens_grids (var0_tdata, 0)
		rval_data, pval_data, sval_data = pazuzu.calc_esa(var0_data, fcst_resp)
		heuristic_data = pazuzu.calc_heuristic(pval_data,pval_threshold,heuristic_radius)
		heuristic_mask = pazuzu.apply_pval_heuristic_filters(pval_data, pval_threshold, heuristic_data, heuristic_multiplier)
		pazuzu.plot_map_outlinebool (rval_data, heuristic_mask, 8, 6, countyshp, stateshp, np.arange(-1.00,1.05,0.05), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/sensitivity_'+case_dir+'_'+case_time+'_'+var1_var+'_'+var0_var+'_'+str(cur_region)+'.png', bounded_region = fcst_resp_mask)
		madis_obs, madis_qcr = pazuzu.read_madis_var(madis_file,var0_var)
		madis_lats = pazuzu.read_madis_var(madis_file,'lat')[0]
		madis_lons = pazuzu.read_madis_var(madis_file,'lon')[0]
		wrf_madis_obs, wrf_madis_x, wrf_madis_y, wrf_madis_lats, wrf_madis_lons, wrf_madis_qc = pazuzu.madis_filter_to_grid_xy(madis_obs, madis_lats, madis_lons, xsize_data, ysize_data, wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_ref_lat, wprj_ref_lon, wprj_known_x, wprj_known_y, wprj_pole_lat, wprj_pole_lon, dx_data, dy_data, edge_filter_width = 10)
		f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, f_wrf_madis_lats, f_wrf_madis_lons = pazuzu.filter_madis_obs(wrf_madis_obs, wrf_madis_x, wrf_madis_y, heuristic_mask, wrf_madis_lats, wrf_madis_lons, wrf_madis_qc, center_x = center_x, center_y = center_y, filter_radius = madis_filter_radius)
		ens_mem_err[i,:] = pazuzu.calc_ens_member_err(f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, var0_data, rval_data, sval_data = sval_data, method = 3, weight_radius = obs_weight_radius)
		unw_ens_pert_mean = pazuzu.calc_ens_mean(var0_data)
		obs_diff, r_obs_diff, s_obs_diff, rs_obs_diff = pazuzu.calc_obs_pert(f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, unw_ens_pert_mean, rval_data, sval_data)
		print(ens_mem_err[i,:])
		pazuzu.plot_basic_scatter(ens_mem_err[i,:], fcst_resp, 8, 6, output_dir+'/weights_'+case_dir+'_'+case_time+'_'+var1_var+'_'+var0_var+'_'+str(cur_region)+'.png', xlabel = "Ensemble Member Error", ylabel = "Forecast Response")
		pazuzu.plot_map_col_obs (rval_data, 8, 6, countyshp, stateshp, np.arange(-1.00,1.05,0.05), xlim_data, ylim_data, lat_data, lon_data, proj_data, s_obs_diff, f_wrf_madis_lats, f_wrf_madis_lons, -4, 4, output_dir+'/obs_'+case_dir+'_'+case_time+'_'+var1_var+'_'+var0_var+'_'+str(cur_region)+'.png', bounded_region = fcst_resp_mask)
	ens_mem_totalerr = np.sum(ens_mem_err,axis=0)
	print(ens_mem_totalerr)
	pazuzu.plot_basic_scatter(ens_mem_totalerr, fcst_resp, 8, 6, output_dir+'/weights_'+case_dir+'_'+case_time+'_'+var1_var+'_'+str(cur_region)+'.png', xlabel = "Total Ensemble Member Error", ylabel = "Forecast Response")
	pazuzu.plot_stacked_bar_chart(ens_mem_err, 8, 6, output_dir+'/weights_allvars_'+case_dir+'_'+case_time+'_'+var1_var+str(cur_region)+'.png', data_labels = var0_labels, ylabel = "Error Contribution")
		
var2_tdata = pazuzu.readensdata_mt(ens_dir, ens_mems, var2_time, var2_var, zaglbottom=var2_zaglbottom, zagltop=var2_zagltop)[0]
var2_data = pazuzu.time_aggregate_ens_grids (var2_tdata, 1)
var2_filtered = pazuzu.ens_neighborhood_filter(var2_data, 1, neighborhood_filter_radius)
var2_exceedance = pazuzu.calc_unw_prob_exceedance (var2_filtered, var2_threshold, 0)
var2_sd = pazuzu.calc_ens_sd(var2_filtered)
pazuzu.plot_map_simple (var2_exceedance, 8, 6, countyshp, stateshp, np.arange(0,1.025,0.025), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/exceedance_'+var2_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var2_sd, 8, 6, countyshp, stateshp, np.arange(0,305,5), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/sd_'+var2_var+'_'+case_dir+'_'+case_time+'.png')
var2_uncertainty = pazuzu.calc_prob_dual_exceedance(var1_exceedance, 0.85, 0, var2_sd, 100, 0, neighborhood_filter_radius)
var2_mean = pazuzu.calc_ens_mean(var2_data)
var2_regions, var2_regions_max = pazuzu.label_regions(var2_uncertainty, 0.5, 0, close_iterations = 3, dilate_iterations = 2, edge_filter = 10)
pazuzu.plot_map_outlinebool (var2_exceedance, var2_mean >= 200, 8, 6, countyshp, stateshp, np.arange(-1.00,1.05,0.05), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/exceedance_overlay_'+var2_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var2_exceedance, 8, 6, countyshp, stateshp, np.arange(0,1.025,0.025), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/exceedance_'+var2_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var2_uncertainty, 8, 6, countyshp, stateshp, np.arange(0,1.025,0.025), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/uncertainty_'+var2_var+'_'+case_dir+'_'+case_time+'.png')	
pazuzu.plot_map_simple (var2_sd, 8, 6, countyshp, stateshp, np.arange(0,305,5), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/sd_'+var2_var+'_'+case_dir+'_'+case_time+'.png')
pazuzu.plot_map_simple (var2_regions, 8, 6, countyshp, stateshp, np.arange(-0.5,var2_regions_max+1.5,1), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/regions_'+var2_var+'_'+case_dir+'_'+case_time+'.png')

for cur_region in np.arange(1,var2_regions_max+1,1):
	center_x, center_y = pazuzu.get_region_center(var1_regions, cur_region)
	fcst_resp_mask = (var2_regions == cur_region)
	fcst_resp = pazuzu.calc_forecastresponse_mt (ens_mems, fcst_resp_mask, var2_tdata, fcst_resp_type = 3, fcst_resp_threshold=var2_threshold, time_operation_type = 1, scl_data=scl_data,dx_data=dx_data,dy_data=dy_data)
	ens_mem_err = np.zeros([len(var0_varlist),ens_mems])
	for i in np.arange(0,len(var0_varlist),1):
		var0_var = var0_varlist[i]
		var0_tdata = pazuzu.readensdata_mt(ens_dir, ens_mems, var0_time, var0_var, zaglbottom=var0_zaglbottom, zagltop=var0_zagltop)[0]
		var0_data = pazuzu.time_aggregate_ens_grids (var0_tdata, 0)
		rval_data, pval_data, sval_data = pazuzu.calc_esa(var0_data, fcst_resp)
		heuristic_data = pazuzu.calc_heuristic(pval_data,pval_threshold,heuristic_radius)
		heuristic_mask = pazuzu.apply_pval_heuristic_filters(pval_data, pval_threshold, heuristic_data, heuristic_multiplier)
		pazuzu.plot_map_outlinebool (rval_data, heuristic_mask, 8, 6, countyshp, stateshp, np.arange(-1.00,1.05,0.05), xlim_data, ylim_data, lat_data, lon_data, proj_data, output_dir+'/sensitivity_'+case_dir+'_'+case_time+'_'+var2_var+'_'+var0_var+'_'+str(cur_region)+'.png', bounded_region = fcst_resp_mask)
		madis_obs, madis_qcr = pazuzu.read_madis_var(madis_file,var0_var)
		madis_lats = pazuzu.read_madis_var(madis_file,'lat')[0]
		madis_lons = pazuzu.read_madis_var(madis_file,'lon')[0]
		wrf_madis_obs, wrf_madis_x, wrf_madis_y, wrf_madis_lats, wrf_madis_lons, wrf_madis_qc = pazuzu.madis_filter_to_grid_xy(madis_obs, madis_lats, madis_lons, xsize_data, ysize_data, wprj_map_proj, wprj_truelat1, wprj_truelat2, wprj_stand_lon, wprj_ref_lat, wprj_ref_lon, wprj_known_x, wprj_known_y, wprj_pole_lat, wprj_pole_lon, dx_data, dy_data, edge_filter_width = 10)
		f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, f_wrf_madis_lats, f_wrf_madis_lons = pazuzu.filter_madis_obs(wrf_madis_obs, wrf_madis_x, wrf_madis_y, heuristic_mask, wrf_madis_lats, wrf_madis_lons, wrf_madis_qc, center_x = center_x, center_y = center_y, filter_radius = madis_filter_radius)
		ens_mem_err[i,:] = pazuzu.calc_ens_member_err(f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, var0_data, rval_data, sval_data = sval_data, method = 3, weight_radius = obs_weight_radius)
		unw_ens_pert_mean = pazuzu.calc_ens_mean(var0_data)
		obs_diff, r_obs_diff, s_obs_diff, rs_obs_diff = pazuzu.calc_obs_pert(f_wrf_madis_obs, f_wrf_madis_x, f_wrf_madis_y, unw_ens_pert_mean, rval_data, sval_data)
		print(ens_mem_err[i,:])
		pazuzu.plot_basic_scatter(ens_mem_err[i,:], fcst_resp, 8, 6, output_dir+'/weights_'+case_dir+'_'+case_time+'_'+var2_var+'_'+var0_var+'_'+str(cur_region)+'.png', xlabel = "Ensemble Member Error", ylabel = "Forecast Response")
		pazuzu.plot_map_col_obs (rval_data, 8, 6, countyshp, stateshp, np.arange(-1.00,1.05,0.05), xlim_data, ylim_data, lat_data, lon_data, proj_data, s_obs_diff, f_wrf_madis_lats, f_wrf_madis_lons, -4, 4, output_dir+'/obs_'+case_dir+'_'+case_time+'_'+var2_var+'_'+var0_var+'_'+str(cur_region)+'.png', bounded_region = fcst_resp_mask)
	ens_mem_totalerr = np.sum(ens_mem_err,axis=0)
	print(ens_mem_totalerr)
	pazuzu.plot_basic_scatter(ens_mem_totalerr, fcst_resp, 8, 6, output_dir+'/weights_'+case_dir+'_'+case_time+'_'+var2_var+'_'+str(cur_region)+'.png', xlabel = "Total Ensemble Member Error", ylabel = "Forecast Response")
	pazuzu.plot_stacked_bar_chart(ens_mem_err, 8, 6, output_dir+'/weights_allvars_'+case_dir+'_'+case_time+'_'+var2_var+str(cur_region)+'.png', data_labels = var0_labels, ylabel = "Error Contribution")
