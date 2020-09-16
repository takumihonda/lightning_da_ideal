import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from tools_LT import read_evar_only, setup_12p, get_rmse_bias3d, get_dat

quick = True
#quick = False


def main( vname1_l, rEXP="2000m_NODA_0306",  EXP1="2000m_DA_0306", EXP2="2000m_DA_0306", time0=datetime(2001,1,1)):

    
    
    tmin_ = 0
    tmax_ = 30
    dt = 5
    
    
    ft_l = np.arange( tmin_, tmax_ + dt, dt ) 
    
    # pcolor mesh
    Z_h = np.arange(DZ*0.5, DZ*(ZDIM+1), DZ) - DZ*0.5
    ft_lh = np.arange( tmin_, tmax_ + 2*dt, dt ) - dt*0.5
    
    y2d, x2d = np.meshgrid( Z_h*0.001, ft_lh )
    
    cmap_rb = plt.cm.get_cmap("RdBu_r")
#    cmap_rb.set_over('gray', alpha=1.0)
#    cmap_rb.set_under('gray', alpha=1.0)
    
    for idx, vname1 in enumerate( vname1_l ):
       INFO["time0"] = time0
       err_t1, err1d_t1, bias_t1, bias1d_t1 = get_dat( INFO, rEXP=rEXP, EXP1=EXP1, vname1=vname1, time0=time0 )
       err_t2, err1d_t2, bias_t2, bias1d_t2 = get_dat( INFO, rEXP=rEXP, EXP1=EXP2, vname1=vname1, time0=time0 )
    

       if vname1 == "QHYD" or vname1 == "QR" or vname1 == "QS" or \
          vname1 == "QG" or vname1 == "QI" or vname1 == "QV":
          unit = "(g/kg)"
          dv = 0.02
          #vmax = 0.13
          #vmin = -0.13
          #levs = np.arange( vmin, vmax+dv, dv ) - dv*0.5
       elif vname1 == "QCRG" or vname1 == "CG" or vname1 == "CS" or \
            vname1 == "CR" or vname1 == "CC" or vname1 == "CI":
          unit = "(nC)"
          dv = 0.01
       elif vname1 == "U" or vname1 == "V" or vname1 == "W":
          unit = "(m/s)"
          dv = 0.1
       elif vname1 == "T":
          unit = "(K)"
          dv = 0.1

       vmax = 5*dv
       vmin = -5*dv
       levs = [ -5*dv, -4*dv, -3*dv, -2*dv,-dv, -0.5*dv,
                0.5*dv, dv, 2*dv, 3*dv, 4*dv, 5*dv]

       fig, ((ax1)) = plt.subplots(1, 1, figsize=(5,4.2))
       fig.subplots_adjust( left=0.15, bottom=0.12, right=0.95, top=0.9,
                            wspace=0.2, hspace=0.2)
    
       ax1.plot( ft_l[:], err_t1, c='k', ls='solid',   label='NO GLMDA')
       ax1.plot( ft_l[:], err_t2, c='r', ls='solid',   label='      GLMDA')
#       ax1.plot( ft_l[:], bias_t1, c='k', ls='dashed', label='Bias   NO GLMDA')
#       ax1.plot( ft_l[:], bias_t2, c='r', ls='dashed', label='Bias      GLMDA')
    
       ax1.legend()
    
#       vmax = np.max( np.abs( err1d_t2 - err1d_t1) )
     
#       norm = BoundaryNorm( levs, ncolors=cmap_rb.N, clip=True )
#
#       print("DEBUG", err_t1)
#       print(x2d.shape, y2d.shape, err1d_t2.shape)
#       SHADE = ax2.pcolormesh( x2d, y2d, err1d_t2 - err1d_t1, 
#                               cmap=cmap_rb,
#                               norm=norm,
#                               vmax=vmax, vmin=vmin, )
#    
#       pos = ax2.get_position()
#       cb_h = pos.height #0.01 #pos.height
#       cb_w = 0.01
#       ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
#    
#       cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical', 
#                          extend='both', ticks=levs[::1] )
#       cb.ax.tick_params( labelsize=8 )
    
       ymin = 0.0
       ymax = 3.0

       ymin = 1.0
       ymax = 2.0

       ax1.set_xlim( tmin_, tmax_)
       ax1.set_ylim( ymin, ymax )

#       ax2.set_xlim( tmin_, tmax_)
#       ax2.set_ylim( 0, 15.0)
    
       ax1.set_xlabel( "Forecast time (min)")
       ax1.set_ylabel( "RMSE")
#       ax2.set_xlabel( "Forecast time (min)")
#       ax2.set_ylabel( "Height (km)")
#    
#       ax2.text( 0.5, 1.01, "RMSE DIF",
#                 fontsize=10, transform=ax2.transAxes,
#                 horizontalalignment='center',
#                 verticalalignment='bottom', )


       #fig.suptitle( "RMSE/Bias: " + vname1 + " " + unit, fontsize=16 )
#       fig.suptitle( "RMSE: " + vname1 + " " + unit, fontsize=14 )
    
       ax1.text( 0.5, 1.01, "RMSE: " + vname1 + " " + unit,
                fontsize=12, transform=ax1.transAxes,
                horizontalalignment='center',
                verticalalignment='bottom', )

       odir = "png/1p_fscore/{0:}_{1:}".format( EXP1, EXP2 )
       ofig = "1p_fscore_" + vname1 + ".png"
    
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
          plt.cla()
          plt.clf()
          plt.close('all')
    






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


EXP = "2000m_DA_0302"


time0 = datetime( 2001, 1, 1, 1, 20, 0 ) 
time0 = datetime( 2001, 1, 1, 1, 30, 0 ) 
time0 = datetime( 2001, 1, 1, 2, 0, 0 ) 

GTOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT"
TYPE = "fcst"
MEM = "mean"
time00 = datetime( 2001, 1, 1, 0, 0, 0 )

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "GTOP":GTOP,
        "ZDIM":ZDIM, "Z":Z, "DT":DT,
        "TYPE":TYPE, "MEM":MEM, "EXP":EXP,
        "time0": time0, "time00": time00  }


EXP1 = "2000m_DA_0306_NOFP"
EXP2 = "2000m_DA_0306_FP_M32_LOC90km"

EXP2 = "2000m_DA_0306_FP_M32_LOC90km_ZMAX23"
#EXP2 = "2000m_DA_0306_FP_M32_LOC90km_HT8"

#EXP2 = "2000m_DA_0306_FP_M32_LOC90km_QC5"
EXP2 = "2000m_DA_0306_FP_M01_LOC90km"
#EXP2 = "2000m_DA_0306_FP_M32_LOC30km"
#EXP2 = "2000m_DA_0306_FP_M32_LOC150km"

rEXP = "2000m_NODA_0306"



EXP1 = "2000m_DA_0723_NOFP"
EXP2 = "2000m_DA_0723_FP"
EXP2 = "2000m_DA_0723_FP_M160"
EXP2 = "2000m_DA_0723_FP_NOB"

EXP2 = "2000m_DA_0723_FP_NOB_OBERR0.1"

EXP1 = "2000m_DA_0723_NOFP_30min"
EXP2 = "2000m_DA_0723_FP_30min_LOC30km"


EXP2 = "2000m_DA_0723_FP_30min_LOC10km_VLOC30km"
EXP2 = "2000m_DA_0723_FP_30min_LOC10km"

rEXP = "2000m_NODA_0723"

vname1_l = [ 
             "W",
            ]

#vname1_l = [ 
#             "T",
#             #"QI",
#            ]
#



main( vname1_l, rEXP=rEXP, EXP1=EXP1, EXP2=EXP2, time0=time0 )

