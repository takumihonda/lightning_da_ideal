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

from matplotlib.colors import BoundaryNorm

from tools_LT import read_evar_only, setup_12p

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

def main( INFO, EXPG="2000m_DA_0306", EXPA="2000m_DA_0306", REXP="2000m_NODA_0601", tlev=0, typ="anal", tit_l=[], vname1="QHYD", vname2="QCRG", zlev_show=1, cx=-1, cy=-1 ):

    print( tlev, INFO["DT"]*tlev )

    #ctime = datetime(2001, 1, 1, 1, 0) + timedelta(seconds=INFO["DT"]*tlev ) 
    ctime = INFO["time0"] + timedelta(seconds=INFO["DT"]*tlev ) 
    if typ is not "fcst":
       ctime = datetime(2001, 1, 1, 1, 0) + timedelta(seconds=INFO["DT"]*tlev ) 

    print( EXPG )
    INFO["EXP"] = EXPG
    INFO["MEM"] = "mean"
    INFO["TYPE"] = typ
    if typ is not "fcst":
       INFO["time0"] = ctime

    print("CHECK", INFO["time0"] )
    tbb_exp1, z_exp1, vr_exp1 = read_vars( INFO, tlev=tlev, HIM8=False )
    evar_exp1 = read_evar_only( INFO, tlev=tlev, vname=vname2 )
    eqh_exp1 = read_evar_only( INFO, tlev=tlev, vname=vname1 ) * 1.e3
    efp_exp1 = read_evar_only( INFO, tlev=tlev, vname="FP" )

    print( "" )
    print( EXPA )
    INFO["EXP"] = EXPA
    tbb_exp2, z_exp2, vr_exp2 = read_vars( INFO, tlev=tlev, HIM8=False )
    evar_exp2 = read_evar_only( INFO, tlev=tlev, vname=vname2 )
    eqh_exp2 = read_evar_only( INFO, tlev=tlev, vname=vname1 ) * 1.e3
    efp_exp2 = read_evar_only( INFO, tlev=tlev, vname="FP" )

    ft_sec =  int( INFO["DT"]*tlev )


    # nature run
    # read variables
    INFO["EXP"] = REXP
    INFO["MEM"] = "mean"
    INFO["TYPE"] = "fcst"
    INFO["time0"] = datetime(2001, 1, 1, 1, 0)
    tlev_nat = int( ( ctime - datetime(2001, 1, 1, 1, 0) ).total_seconds() / INFO["DT"] )
    print( "DEBUG", tlev_nat, ctime)
    tbb_nat, z_nat, vr_nat = read_vars( INFO, tlev=tlev_nat, HIM8=False )
    evar_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname2 )
    efp_nat = read_evar_only( INFO, tlev=tlev_nat, vname="FP" )
    ew_nat = read_evar_only( INFO, tlev=tlev_nat, vname="W" )
    qh_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname1 ) * 1.e3
 
    if vname2 == "W" or vname2 == "V":
       eqh_exp1 = eqh_exp1 / 1.e3
       eqh_exp2 = eqh_exp2 / 1.e3
       qh_nat = qh_nat / 1.e3


    print("evars: ", evar_nat.shape, evar_exp1.shape, evar_exp2.shape )


    if typ is "fcst":
       foot = "\n(fcst from mean)"
       if ft_sec == 0:
          foot = "\n(analysis)"
       foot = "" # DEBUG
       tit_l_ = [
                 tit_l[0] + foot, 
                 tit_l[1] + foot, 
                 tit_l[2],
                 tit_l[0] + foot, 
                 tit_l[1] + foot, 
                 tit_l[2],
                 "",
                 ]
    else:
       foot = ""
       tit_l_ = [
                 tit_l[0] + foot, 
                 tit_l[1] + foot, 
                 tit_l[2],
                 tit_l[0] + foot, 
                 tit_l[1] + foot, 
                 tit_l[2],
                 ]

    print( z_nat.shape, z_exp1.shape, z_exp2.shape )





    ax_l, crs_l, fig =  setup_12p()


    levs_dbz= np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    cmap_dbz = mcolors.ListedColormap(['cyan','dodgerblue', 
                                       'lime', 'limegreen','yellow',
                                       'orange', 'red', 'firebrick', 'magenta',
                                       'purple'])
    cmap_dbz.set_under('w', alpha=1.0)
    cmap_dbz.set_over('gray', alpha=1.0)

    cmap_rb = plt.cm.get_cmap("RdBu_r")
    cmap_rb.set_under('gray', alpha=1.0)
    cmap_rb.set_over('k', alpha=1.0)


    unit_dbz = "(dBZ)"
    unit_crg = "(nC)"
 
    levs_rb_qcrg_ainc = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.01,
                                  0.01, 0.05, 0.1, 0.2, 0.3, 0.4, ])

    if vname2 == "QCRG" or vname2 == "CR":
       levs_rb_qcrg = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.01,
                               0.01, 0.05, 0.1, 0.2, 0.3, 0.4, ])
    elif vname2 == "W" or vname2 == "V":
       levs_rb_qcrg = np.array([-75, -60, -45, -30, -15, -5,
                               5, 15, 30, 45, 60, 75, ])
       levs_rb_qcrg_ainc = np.array([ -10, -8, -6, -4, -2, -1,
                                       1, 2, 4, 6, 8, 10, ] )
    else:
       levs_rb_qcrg = np.array([ -2.4, -2.0, -1.6, -1.2, -0.8, -0.4,  
                                 0.4, 0.8, 1.2, 1.6, 2, 2.4])


    levs_dbz_ai = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, -0.1,
                             0.1, 0.2,  0.4, 0.6,0.8, 1.0])
 
    cmap_dbz = mcolors.ListedColormap(['cyan','dodgerblue',
                                       'lime', 'limegreen','yellow',
                                       'orange', 'red', 'firebrick', 'magenta',
                                       'purple'])
    cmap_dbz.set_under('w', alpha=1.0)
    cmap_dbz.set_over('gray', alpha=1.0)
       

    if vname1 == "QHYD":
       levs_dbz = np.array([0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    elif vname1 == "V" or vname1 == "U":
       levs_dbz = np.array([-75, -60, -45, -30, -15, -5,
                               5, 15, 30, 45, 60, 75, ])
       levs_dbz_ai = np.array([ -10, -8, -6, -4, -2, -1,
                                  1, 2, 4, 6, 8, 10, ] )
       cmap_dbz = cmap_rb
    else:
       levs_dbz = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10])



#    levs_l = [ levs_dbz, levs_dbz_ai, levs_dbz, 
#               levs_rb_qcrg, levs_rb_qcrg_ainc, levs_rb_qcrg,
#               levs_dbz, levs_dbz, levs_dbz_ai, levs_dbz_ai, levs_dbz, levs_dbz,
#               levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg_ainc, levs_rb_qcrg_ainc, levs_rb_qcrg, levs_rb_qcrg, ]
#    cmap_l = [ cmap_dbz, cmap_rb, cmap_dbz, 
#               cmap_rb, cmap_rb, cmap_rb,
#               cmap_dbz, cmap_dbz, cmap_rb, cmap_rb, cmap_dbz, cmap_dbz,
#               cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb ]

    levs_l = [ levs_dbz, levs_dbz_ai, levs_dbz_ai, 
               levs_rb_qcrg, levs_rb_qcrg_ainc, levs_rb_qcrg_ainc,
               levs_dbz, levs_dbz, levs_dbz_ai, levs_dbz_ai, levs_dbz_ai, levs_dbz_ai,
               levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg_ainc, levs_rb_qcrg_ainc, levs_rb_qcrg_ainc, levs_rb_qcrg_ainc, ]

    cmap_l = [ cmap_dbz, cmap_rb, cmap_rb, 
               cmap_rb, cmap_rb, cmap_rb,
               cmap_dbz, cmap_dbz, cmap_rb, cmap_rb, cmap_rb, cmap_rb,
               cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb ]
    unit_l = [ unit_dbz, unit_dbz, unit_dbz,
               unit_crg, unit_crg, unit_crg,
               unit_dbz, unit_dbz, unit_dbz,
               unit_crg, unit_crg, unit_crg ]

    pnum_l = [
              "(a)", "(b)", "(c)",
              "(d)", "(e)", "(f)",
              ]

    tvar = vname2
    if vname2 is "QCRG":
       levs = levs_rb_qcrg
       cmap = cmap_rb
       unit = unit_crg
       tvar = "Total charge density"


    bbox = { 'facecolor':'w', 'alpha':0.95, 'pad':1.5, 'edgecolor':'w' }


    xmin = 120
    xmax = 280
    ymin = 120
    ymax = 280
    zmin = 0.0
    zmax = 15.0

    ft_sec_a =  int( ( ctime - INFO["time00"] ).total_seconds() )
    print( "ctime",ctime, tlev, INFO["DT"])

   
    xlabel = "X (km)"
    ylabel = "Y (km)"
    zlabel = "Z (km)"

    xaxis = INFO["X"][:] * 0.001
    yaxis = INFO["Y"][:] * 0.001

    x2d, y2d = np.meshgrid( yaxis, xaxis )
    xdgrid = 20
    ydgrid = 20


    #cy, cx = np.unravel_index( np.argmax(ew_nat[0,zlev_show,:,:]), ew_nat[0,0,:,:].shape)


    if cx < 0 or cy < 0: 
       cy, cx = np.unravel_index( np.argmax(z_nat[zlev_show,:,:]), ew_nat[0,0,:,:].shape)

       cy = 102
       cx = 86

    #cx = 76
    #cy = 89

    print("CX,CY:", cx, cy)

    #cx = 98
    #cy = 106

    #cx = 100
    #cy = 111

    #cx = 84
    #cy = 95

    #cx = 90
    #cy = 93

    if typ is not "fcst":
       info = 't={0:.0f}min\nZ={1:}km'.format( ft_sec_a/60, INFO["Z"][zlev_show]/1000)
    else:
       info = 't={0:.0f}min\n(FT={1:.0f}min)\nZ={2:}km'.format( ft_sec_a/60, ft_sec/60, INFO["Z"][zlev_show]/1000)
    

    if typ != "fcst":
       VAR_l = [ 
                 z_exp1[zlev_show,:,:], 
                 z_exp2[zlev_show,:,:], 
                 z_nat[zlev_show,:,:],
                 evar_exp1[0,zlev_show,:,:], 
                 evar_exp2[0,zlev_show,:,:], 
                 evar_nat[0,zlev_show,:,:],
                 np.transpose( z_exp1[:,:,cx] ), 
                 np.transpose( z_exp2[:,:,cx] ), 
                 np.transpose( z_nat[:,:,cx] ), ]
    else:
       VAR_l = [ 
                 eqh_exp1[0,zlev_show,:,:], 
                 eqh_exp2[0,zlev_show,:,:] - eqh_exp1[0,zlev_show,:,:], 
                 #qh_nat[0,zlev_show,:,:], 
                 qh_nat[0,zlev_show,:,:] - eqh_exp1[0,zlev_show,:,:], # T-B 
                 evar_exp1[0,zlev_show,:,:], 
                 evar_exp2[0,zlev_show,:,:] - evar_exp1[0,zlev_show,:,:], 
                 #evar_nat[0,zlev_show,:,:],
                 evar_nat[0,zlev_show,:,:] - evar_exp1[0,zlev_show,:,:], # T-B
                 np.transpose( eqh_exp1[0,:,:,cx] ), 
                 eqh_exp1[0,:,cy,:], 
                 np.transpose( eqh_exp2[0,:,:,cx] ) - np.transpose( eqh_exp1[0,:,:,cx] ), 
                 eqh_exp2[0,:,cy,:] - eqh_exp1[0,:,cy,:], 
                 #np.transpose( qh_nat[0,:,:,cx] ), 
                 np.transpose( qh_nat[0,:,:,cx] ) - np.transpose( eqh_exp1[0,:,:,cx] ), # T-B 
                 #qh_nat[0,:,cy,:], 
                 qh_nat[0,:,cy,:] - eqh_exp1[0,:,cy,:], # T-B 
                 np.transpose( evar_exp1[0,:,:,cx]), 
                 evar_exp1[0,:,cy,:], 
                 np.transpose( evar_exp2[0,:,:,cx] ) - np.transpose( evar_exp1[0,:,:,cx] ), 
                 evar_exp2[0,:,cy,:] - evar_exp1[0,:,cy,:], 
                 #np.transpose( evar_nat[0,:,:,cx] ),
                 np.transpose( evar_nat[0,:,:,cx] ) - np.transpose( evar_exp1[0,:,:,cx] ), # T-B
                 #evar_nat[0,:,cy,:],
                 evar_nat[0,:,cy,:] - evar_exp1[0,:,cy,:], # T-B
                ]
       FP_l = [ np.sum( efp_exp1[0,:,:,:], axis=0 ), 
                np.sum( efp_exp2[0,:,:,:], axis=0 ), 
                np.sum( efp_nat[0,:,:,:], axis=0 ),
                np.sum( efp_exp1[0,:,:,:], axis=0 ),
                np.sum( efp_exp2[0,:,:,:], axis=0 ),
                np.sum( efp_nat[0,:,:,:], axis=0 ), 
               ]


    inf = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/" + EXPA + "/loc.txt"
    print( inf )
    #loc_data = np.loadtxt( inf, delimiter=",", dtype='int')

    for idx, ax in enumerate(ax_l):
       print("DEBUG", idx, crs_l[idx]) 

       xdgrid_ = xdgrid
       ydgrid_ = ydgrid

       xmin_ = xmin
       ymin_ = ymin
       xmax_ = xmax
       ymax_ = ymax


       if crs_l[idx] == "ZY":
          xaxis = INFO["Z"][:] * 0.001
          yaxis = INFO["Y"][:] * 0.001
          x2d, y2d = np.meshgrid( yaxis, xaxis )

          ymin_ = zmin
          ymax_ = zmax

          ax.hlines( y=INFO["Z"][zlev_show]*0.001, xmin=xmin_, xmax=xmax_,
                     colors="k",linestyles='dotted',linewidths=1.0 )
          ax.vlines( x=INFO["X"][cx]*0.001, ymin=ymin_, ymax=ymax_,
                     colors="k",linestyles='dotted',linewidths=1.0 )

          ydgrid_ = 2
          xdgrid_ = 20

       elif crs_l[idx] == "XZ":
          xaxis = INFO["Y"][:] * 0.001
          yaxis = INFO["Z"][:] * 0.001
          x2d, y2d = np.meshgrid( yaxis, xaxis )

          xmin_ = zmin
          xmax_ = zmax

          ax.hlines( y=INFO["Y"][cy]*0.001, xmin=xmin_, xmax=xmax_,
                     colors="k",linestyles='dotted',linewidths=1.0 )
          ax.vlines( x=INFO["Z"][zlev_show]*0.001, ymin=ymin, ymax=ymax,
                     colors="k",linestyles='dotted',linewidths=1.0 )


          xdgrid_ = 2
          ydgrid_ = 20

       elif crs_l[idx] == "XY":
          ax.vlines( x=INFO["X"][cx]*0.001, ymin=ymin, ymax=ymax,
                     colors="k",linestyles='dotted',linewidths=1.0 )
          ax.hlines( y=INFO["Y"][cy]*0.001, xmin=xmin, xmax=xmax,
                     colors="k",linestyles='dotted',linewidths=1.0 )

       print( VAR_l[idx].shape, x2d.shape, np.max(VAR_l[idx]))

       norm = BoundaryNorm(levs_l[idx], ncolors=cmap_l[idx].N, clip=True)
       #SHADE = ax.pcolormesh(x2d, y2d,
       SHADE = ax.contourf(x2d, y2d,
                           VAR_l[idx][:,:],
                           levels=levs_l[idx],
                           #vmin=np.min(levs),
                           #vmax=np.max(levs),
                           cmap=cmap_l[idx],
                           extend='both',
                           norm=norm,
                           )

#       if idx == 1 or idx == 4:
#          ax.scatter( INFO["X"][loc_data[:,0]-1]*0.001, 
#                      INFO["Y"][loc_data[:,1]-1]*0.001,  
#                      marker='s', s=5, linewidths=0.3,
#                      edgecolors='k', facecolors="None", alpha=1.0,
#                   )

       if typ is "fcst" and ft_sec > 0 and idx <= 5:
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

       ax.set_xlim( xmin_, xmax_ )
       ax.set_ylim( ymin_, ymax_ )

       ax.xaxis.set_ticks( np.arange(xmin_, xmax_, xdgrid_) )
       ax.yaxis.set_ticks( np.arange(ymin_, ymax_, ydgrid_) )
       ax.tick_params(axis='both', which='minor', labelsize=6 )
       ax.tick_params(axis='both', which='major', labelsize=6 )
       
       if idx <= 5:
          ax.text(0.5, 0.95, tit_l_[idx],
                  fontsize=12, transform=ax.transAxes,
                  horizontalalignment='center',
                  verticalalignment='top', 
                  bbox=bbox )

          ax.text(0.1, 0.95, pnum_l[idx],
                  fontsize=10, transform=ax.transAxes,
                  horizontalalignment='center',
                  verticalalignment='top', 
                  bbox=bbox )


       xlabel_ = xlabel
       ylabel_ = ylabel
       if crs_l[idx] == "XZ":
          ylabel_ = ""
          xlabel_ = zlabel
       elif crs_l[idx] == "ZY":
          xlabel_ = ""
          ylabel_ = zlabel

       ax.set_xlabel( xlabel_, fontsize=6 )
       ax.set_ylabel( ylabel_, fontsize=6 )

       if idx <= 5:
   
          pos = ax.get_position()
          #cb_h = pos.height
          #cb_w = 0.01
          cb_h = 0.01
          cb_w = pos.width * 1.5
          ax_cb = fig.add_axes( [pos.x0, pos.y0-0.06, cb_w, cb_h] )
          cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'horizontal', 
                             ticks=levs_l[idx], extend='both' )
          cb.ax.tick_params( labelsize=6 )
          ax.text( 1.15, -0.12, unit_l[idx],
                   fontsize=8, transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top', )
      
       if idx == 2 or idx == 5:
          ax.text( 1.1, 1.2, info,
                   fontsize=9, transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='bottom', )

       if idx == 9 or idx == 15:
          tvar_ = tvar
          if idx == 9:
             if vname1 == "QHYD":
                tvar_ = "Total hydrometeor"
             else:
                tvar_ = vname1
          ax.text( 0.5, 1.15, tvar_,
                   fontsize=13, transform=ax.transAxes,
                   horizontalalignment='center',
                   verticalalignment='center', )

#    fig_tit =  tvar
#    fig.suptitle( fig_tit, fontsize=16 )


    #odir = 'png/18p_DA_ainc/{0:}/i{1:03}_j{2:03}'.format( EXPA, cx, cy )
    odir = 'png/18p_DA_ainc/{0:}'.format( EXPA,  )

    ofig = '18p_ainc_{0:}_{1:}_{2:}_fta{3:05}_ft{4:05}_z{5:0=2}_{6:}_{7:}.png'.format(typ, EXPG, EXPA, ft_sec_a, ft_sec, zlev_show, vname1, vname2)


    print( ofig, odir )
 
    if not quick:
       os.makedirs(odir, exist_ok=True)
       plt.savefig(os.path.join(odir,ofig),
                   bbox_inches="tight", pad_inches = 0.1)
       plt.cla()
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


time0 = datetime( 2001, 1, 1, 1, 20, 0 ) 
time0 = datetime( 2001, 1, 1, 1, 30, 0 ) 
time0 = datetime( 2001, 1, 1, 2,  0, 0 ) 
#time0 = datetime( 2001, 1, 1, 1, 40, 0 ) 

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

#tmin = 6
tmax = tmin + 1
tmax = 7
tmax = 1

typ = "anal"
typ = "fcst"

vname = "QCRG"

tit_l = ["NODA", "DA", "Nature run"]




EXPA = "2000m_DA_0723_FP_30min"
EXPA = "2000m_DA_0723_FP_30min_M64"
EXPA = "2000m_DA_0723_FP_30min_LOC30km" 

EXPA = "2000m_DA_0723_FP_30min_LOC10km_HT16"
EXPA = "2000m_DA_0723_FP_30min_LOC30km_X175km_Y183km"

EXPA = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km_LOG"

EXPA = "2000m_DA_0723_FP_30min_LOC90km_LOC2D"


obsx = 176.0
obsy = 223.0

EXPA = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km"
obsx = 175.0
obsy = 183.0

#EXPA = "2000m_DA_0723_FP_30min_LOC10km_X167km_Y223km"
#obsx = 167.0
#obsy = 223.0

EXPA = "2000m_DA_0723_FP_30min_LOC10km"
EXPA = "2000m_DA_0723_FP_30min_LOC30km"
#EXPA = "2000m_DA_0723_FP_30min_LOC10km_LOG"
#EXPA = "2000m_DA_0723_FP_30min_LOC10km_VLOCW20"

EXPA = "2000m_DA_0723_FP_30min_LOC30km_X183km_Y199km"
EXPA = "2000m_DA_0723_Z20km_FP_30min_LOC30km"

EXPA = "2000m_DA_0723_FP_30min_LOC30km_COERR0.2"
EXPA = "2000m_DA_0723_FP_30min_LOC20km"

obsx = 183.0
obsy = 199.0

EXPA = "2000m_DA_0723_FP_30min_LOC20km_X159km_Y231km"
EXPA = "2000m_DA_0723_FP_30min_LOC20km_M240"

EXPA = "2000m_DA_0723_FP_30min_LOC20km_M240_NOG"
EXPA="2000m_DA_0723_FP_30min_LOC20km_M240_NOCLROBS"
obsx = 159.0
obsy = 231.0


EXPG = "2000m_DA_0723_NOFP_30min"

REXP = "2000m_NODA_0723"


tit_l = ["GUESS", "ANAL INC", "Nature run"]
tit_l = ["GUESS", "ANAL INC", "Truth-B"]


vname1 = "QR"
vname2 = "CR"
#vname1 = "QG"
#vname2 = "CG"
#vname1 = "QS"
#vname2 = "CS"
#vname1 = "QHYD"
#vname2 = "QCRG"

vname1_l = [ "V", "QR",
             "QG",
             "QS",
             "QHYD",
            ]

vname2_l = [ "W", "CR",
             "CG",
             "CS",
             "QCRG",
            ]

vname1_l = [ "QG" ]
vname2_l = [ "CG" ]

zlev_min = 10
zlev_max = 28
dz = 4

cx = np.argmin( np.abs( INFO["X"][:]*0.001 - obsx ) )
cy = np.argmin( np.abs( INFO["Y"][:]*0.001 - obsy ) )
print( cx, cy )

if typ is not "fcst":
   tmin = 1

for tlev in range( tmin, tmax ):
    for zlev_show in range( zlev_min, zlev_max+dz, dz):
        for idx, vname1 in enumerate(vname1_l):
            INFO["time0"] = time0
            main( INFO, EXPG=EXPG, EXPA=EXPA, REXP=REXP, tlev=tlev, typ=typ, tit_l=tit_l, vname1=vname1, vname2=vname2_l[idx], zlev_show=zlev_show, cx=cx, cy=cy )
