import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta

import os
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from tools_LT import read_evar_only

quick = True
quick = False

def read_vars( INFO, tlev=0, HIM8=True ):

    # Read variables
    if HIM8:
       fn_Him8 = os.path.join( INFO["GTOP"], INFO["EXP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], INFO["MEM"], 
                               "Him8_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".nc") 
       print( fn_Him8 )
       nc = Dataset(fn_Him8, 'r', format='NETCDF4')
       tbb = nc.variables["tbb"][tlev,:,:,:]
       nc.close()
    else:
       tbb = np.zeros(1)

    fn_radar = os.path.join( INFO["GTOP"], INFO["EXP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], INFO["MEM"], 
                       "radar_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".nc") 
    print( fn_radar, tlev )
    nc = Dataset(fn_radar, 'r', format='NETCDF4')
    if INFO["TYPE"] is "fcst":
       z = nc.variables["z"][tlev,:,:,:]
       vr = nc.variables["vr"][tlev,:,:,:]
    else:
       z = nc.variables["z"][:,:,:]
       vr = nc.variables["vr"][:,:,:]
    nc.close()

    return( tbb, z, vr )

def main( INFO, EXP1="2000m_DA_0306", EXP2="2000m_DA_0306", tlev=0, typ="anal", vname="QG" ):

    print( tlev, INFO["DT"]*tlev )

    #ctime = datetime(2001, 1, 1, 1, 0) + timedelta(seconds=INFO["DT"]*tlev ) 
    ctime = INFO["time0"] + timedelta(seconds=INFO["DT"]*tlev ) 
    if typ is not "fcst":
       ctime = datetime(2001, 1, 1, 1, 0) + timedelta(seconds=INFO["DT"]*tlev ) 

    INFO["EXP"] = EXP1
    INFO["MEM"] = "mean"
    INFO["TYPE"] = typ
    if typ is not "fcst":
       INFO["time0"] = ctime

    print("CHECK", INFO["time0"] )
    tbb_exp1, z_exp1, vr_exp1 = read_vars( INFO, tlev=tlev, HIM8=False )
    evar_exp1 = read_evar_only( INFO, tlev=tlev, vname=vname )
    efp_exp1 = read_evar_only( INFO, tlev=tlev, vname="FP" )

    INFO["EXP"] = EXP2
    tbb_exp2, z_exp2, vr_exp2 = read_vars( INFO, tlev=tlev, HIM8=False )
    evar_exp2 = read_evar_only( INFO, tlev=tlev, vname=vname )
    efp_exp2 = read_evar_only( INFO, tlev=tlev, vname="FP" )

    ft_sec =  int( INFO["DT"]*tlev )

    # nature run
    # read variables
    INFO["EXP"] = EXP1
    INFO["MEM"] = "mean"
    INFO["TYPE"] = "fcst"
    INFO["time0"] = datetime(2001, 1, 1, 1, 0)
    tlev_nat = int( ( ctime - datetime(2001, 1, 1, 1, 0) ).total_seconds() / INFO["DT"] )
    print( "DEBUG", tlev_nat, ctime)
    tbb_nat, z_nat, vr_nat = read_vars( INFO, tlev=tlev_nat, HIM8=False )
    evar_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname )
    efp_nat = read_evar_only( INFO, tlev=tlev_nat, vname="FP" )
 
    print("evars: ", evar_nat.shape, evar_exp1.shape, evar_exp2.shape )


    tit_l = [ "NODA (analysis)", 
              "DA (analysis)", 
              "Nature run" ]
    if typ is "fcst":
       foot = "\n(fcst from mean)"
       if ft_sec == 0:
          foot = "\n(analysis)"
       tit_l = [
                 "NODA" + foot, 
                 "DA" + foot, 
                 "Nature run",
                 "NODA" + foot, 
                 "DA" + foot, 
                 "Nature run",
                 ]

    print( z_nat.shape, z_exp1.shape, z_exp2.shape )


    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6) ) = plt.subplots(2, 3, figsize=(11,8.2))
    fig.subplots_adjust(left=0.06, bottom=0.05, right=0.93, top=0.94,
                        wspace=0.2, hspace=0.3)
    
    ax_l = [ax1, ax2, ax3, ax4, ax5, ax6]

    levs_dbz= np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    cmap_dbz = mcolors.ListedColormap(['cyan','dodgerblue', 
                                       'lime', 'limegreen','yellow',
                                       'orange', 'red', 'firebrick', 'magenta',
                                       'purple'])
    cmap_dbz.set_under('w', alpha=1.0)
    cmap_dbz.set_over('gray', alpha=1.0)

    cmap_rb = plt.cm.get_cmap("RdBu_r")
    cmap_rb.set_under('gray', alpha=1.0)
    cmap_rb.set_over('gray', alpha=1.0)


    unit_dbz = "(dBZ)"
    unit_crg = r'(nC m$^{-3}$)'
 
    #levs_rb_qcrg = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1,
    #                        0.1, 0.2, 0.4, 0.6, 0.8, 1])
    levs_rb_qcrg = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.01,
                            0.01, 0.05, 0.1, 0.2, 0.3, 0.4, ])


    levs_rb_qcrg = np.array([-0.6, -0.4, -0.2, -0.1, -0.05, -0.01,
                              0.01, 0.05, 0.1, 0.2, 0.4, 0.6])

    levs_l = [ levs_dbz, levs_dbz, levs_dbz, 
               levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg]
    cmap_l = [ cmap_dbz, cmap_dbz, cmap_dbz, 
               cmap_rb, cmap_rb, cmap_rb ]
    unit_l = [ unit_dbz, unit_dbz, unit_dbz,
               unit_crg, unit_crg, unit_crg ]

    pnum_l = [
              "(a)", "(b)", "(c)",
              "(d)", "(e)", "(f)",
              ]

    tvar = vname
    if vname is "QCRG":
       levs = levs_rb_qcrg
       cmap = cmap_rb
       unit = unit_crg
       tvar = "Total charge density"


    bbox = { 'facecolor':'w', 'alpha':0.95, 'pad':1.5, 'edgecolor':'w' }


    xmin = 120
    xmax = 280
    ymin = 120
    ymax = 320


    ft_sec_a =  int( ( ctime - INFO["time00"] ).total_seconds() )
    print( "ctime",ctime, tlev, INFO["DT"])

   
    xlabel = "X (km)"
    ylabel = "Y (km)"

    xaxis = INFO["X"][:] * 0.001
    yaxis = INFO["Y"][:] * 0.001

    x2d, y2d = np.meshgrid( yaxis, xaxis )
    xdgrid = 20
    ydgrid = 20

    zlev_show = 8 
    zlev_show = 10
    zlev_show = 16
    zlev_show = 14 # comment out 

    if typ is not "fcst":
       info = 't={0:.0f} min\nZ={1:} km'.format( ft_sec_a/60.0, INFO["Z"][zlev_show]/1000)
    else:
       info = 't={0:.0f} min (FT={1:.0f} min)\nZ={2:} km'.format( ft_sec_a/60.0, ft_sec/60.0, INFO["Z"][zlev_show]/1000)
    
    if typ is not "fcst":
       VAR_l = [ 
                 z_exp1[zlev_show,:,:], 
                 z_exp2[zlev_show,:,:], 
                 z_nat[zlev_show,:,:],
                 evar_exp1[0,zlev_show,:,:], 
                 evar_exp2[0,zlev_show,:,:], 
                 evar_nat[0,zlev_show,:,:]]
    else:
       VAR_l = [ 
                 z_exp1[zlev_show,:,:], 
                 z_exp2[zlev_show,:,:], 
                 z_nat[zlev_show,:,:], 
                 evar_exp1[0,zlev_show,:,:], 
                 evar_exp2[0,zlev_show,:,:], 
                 evar_nat[0,zlev_show,:,:]
                ]
       FP_l = [ np.sum( efp_exp1[0,:,:,:], axis=0), 
                np.sum(efp_exp2[0,:,:,:], axis=0), 
                np.sum(efp_nat[0,:,:,:], axis=0) ]

    for idx, ax in enumerate(ax_l):
#       print(idx,tit_l[idx])
       print( VAR_l[idx].shape, np.max(VAR_l[idx]), np.min(VAR_l[idx]) )

       #SHADE = ax.pcolormesh(x2d, y2d,
       SHADE = ax.contourf(x2d, y2d,
                           VAR_l[idx][:,:],
                           levels=levs_l[idx],
                           #vmin=np.min(levs),
                           #vmax=np.max(levs),
                           cmap=cmap_l[idx],
                           extend='both',
                           )

       if typ is "fcst" and ft_sec > 0:
         ssize = 10.0
         idx_ = idx
         if idx > 2:
            idx_ = idx - 3
         fp2d = FP_l[idx_] 
         #fp2d[ fp2d < 1.0 ] = np.nan
         #fp2d = fp2d / ssize
         fp2d = np.where( fp2d >= 1.0, ssize, np.nan )
         ax.scatter( x2d, y2d, s=fp2d, 
                     c='k', marker='s', 
                     edgecolors="w", linewidths=0.5 )

       ax.set_xlim( xmin, xmax )
       ax.set_ylim( ymin, ymax )
       ax.xaxis.set_ticks( np.arange(xmin, xmax, xdgrid) )
       ax.yaxis.set_ticks( np.arange(ymin, ymax, ydgrid) )
       ax.tick_params(axis='both', which='minor', labelsize=7 )
       ax.tick_params(axis='both', which='major', labelsize=7 )

       ax.text(0.5, 0.95, tit_l[idx],
               fontsize=12, transform=ax.transAxes,
               horizontalalignment='center',
               verticalalignment='top', 
               bbox=bbox )
       
       ax.text(0.1, 0.95, pnum_l[idx],
               fontsize=10, transform=ax.transAxes,
               horizontalalignment='center',
               verticalalignment='top', 
               bbox=bbox )

       ax.set_xlabel( xlabel, fontsize=8 )
       ax.set_ylabel( ylabel, fontsize=8 )

       if idx == 2 or idx == 5:
   
          pos = ax.get_position()
          cb_h = pos.height
          cb_w = 0.01
          ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
          cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical', 
                             ticks=levs_l[idx], extend='both' )
          cb.ax.tick_params( labelsize=8 )
          ax.text( 1.0, -0.03, unit_l[idx],
                   fontsize=9, transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='top', )
      
          ax.text( 1.0, 1.1, info,
                   fontsize=10, transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='center', )

       if idx == 1 or idx == 4:
          tvar_ = tvar
          if idx == 1:
             tvar_ = "Radar reflectivity"
          ax.text( 0.5, 1.1, tvar_,
                   fontsize=15, transform=ax.transAxes,
                   horizontalalignment='center',
                   verticalalignment='center', )

#    fig_tit =  tvar
#    fig.suptitle( fig_tit, fontsize=16 )


    #odir = "png/6p_DA_var" 
    odir = "pdf/fig20210624" 

    #ofig = '6p_{:1}_{:2}_{:3}_fta{:05}_ft{:05}_z{:0=2}_{:}.png'.format(typ, EXP1, EXP2, ft_sec_a, ft_sec, zlev_show, vname)
    ofig = '6p_{:1}_{:2}_{:3}_fta{:05}_ft{:05}_z{:0=2}_{:}.pdf'.format(typ, EXP1, EXP2, ft_sec_a, ft_sec, zlev_show, vname)


    print( ofig, odir )
 
    if not quick:
       os.makedirs(odir, exist_ok=True)
       plt.savefig(os.path.join(odir,ofig),
                   bbox_inches="tight", pad_inches = 0.1)
       plt.clf()
       plt.close('all')
    else:
       plt.show()



###################

DX = 2000.0
DY = 2000.0
XDIM = 192
YDIM = 192
TDIM = 13
ZDIM = 40

XDIM = 176
YDIM = 176
ZDIM = 45

DZ = 500.0
DT = 300

X = np.arange( DX*0.5, DX*XDIM, DX )
Y = np.arange( DY*0.5, DY*YDIM, DY )
T = np.arange( 0, DT*TDIM, DT )
BAND = np.arange( 7, 17, 1 )

Z = np.arange(DZ*0.5, DZ*ZDIM, DZ)

#EXP = "2000m_NODA_1022_FIR2km_N"
#time0 = datetime( 2001, 1, 1, 1, 0, 0 )
EXP = "2000m_DA_1022_FIR2km_N"

EXP = "2000m_DA_0302"

EXP1 = "2000m_DA_0306"

EXP1 = "2000m_NODA_0306"
EXP2 = "2000m_DA_0306"


EXP1 = "2000m_NODA_0601"
EXP2 = "2000m_DA_0601"

EXP1 = "2000m_NODA_0723"
EXP2 = "2000m_DA_0723"

#EXP1 = "2000m_DA_0306_R_FP_180km"

time0 = datetime( 2001, 1, 1, 1, 20, 0 ) 
time0 = datetime( 2001, 1, 1, 1, 30, 0 ) 

GTOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT"
TYPE = "fcst"
MEM = "mean"
MEM = "0025"
time00 = datetime( 2001, 1, 1, 0, 0, 0 )

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "GTOP":GTOP,
        "ZDIM":ZDIM, "Z":Z, "DT":DT,
        "TYPE":TYPE, "MEM":MEM, "EXP":EXP,
        "time0": time0, "time00": time00  }

tmax = 13
tmax = 7
tmin = 0

tmin = 0
tmax = tmin + 1
#tmin = 6
tmax = 7
#tmax = 1

typ = "anal"
typ = "fcst"

vname = "QCRG"


if typ is not "fcst":
   tmin = 1

for tlev in range( tmin, tmax ):
    INFO["time0"] = time0
    main( INFO, EXP1=EXP1, EXP2=EXP2, tlev=tlev, typ=typ, vname=vname )
