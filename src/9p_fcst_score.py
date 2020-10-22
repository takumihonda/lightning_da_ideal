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

from tools_LT import read_evar_only, setup_9p, get_dat

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

def main( INFO, EXP1="2000m_DA_0306", EXP2="2000m_DA_0306", NEXP="2000m_NODA_0306", tlev=0, typ="anal", tit_l=[], vname1="QHYD", vname2="QHYD", vname3="W", vname4="W", zlev_show=1, time0=datetime(2001,1,1) ):

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
    evar_exp1 = read_evar_only( INFO, tlev=tlev, vname=vname1 ) 
    efp_exp1 = read_evar_only( INFO, tlev=tlev, vname="FP" )

    INFO["EXP"] = EXP2
    tbb_exp2, z_exp2, vr_exp2 = read_vars( INFO, tlev=tlev, HIM8=False )
    evar_exp2 = read_evar_only( INFO, tlev=tlev, vname=vname1 )
    efp_exp2 = read_evar_only( INFO, tlev=tlev, vname="FP" )

    ft_sec =  int( INFO["DT"]*tlev )


    # nature run
    # read variables
    INFO["EXP"] = NEXP
    INFO["MEM"] = "mean"
    INFO["TYPE"] = "fcst"
    INFO["time0"] = datetime(2001, 1, 1, 1, 0)
    tlev_nat = int( ( ctime - datetime(2001, 1, 1, 1, 0) ).total_seconds() / INFO["DT"] )
    print( "DEBUG", tlev_nat, ctime)
    tbb_nat, z_nat, vr_nat = read_vars( INFO, tlev=tlev_nat, HIM8=False )
    evar_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname1 )
    efp_nat = read_evar_only( INFO, tlev=tlev_nat, vname="FP" )
    ew_nat = read_evar_only( INFO, tlev=tlev_nat, vname="W" )
    qh_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname1 ) 
 



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



    fac = 1.0

    ax_l, crs_l, fig =  setup_9p()


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

    cmap_dbz = mcolors.ListedColormap(['cyan','dodgerblue',
                                       'lime', 'limegreen','yellow',
                                       'orange', 'red', 'firebrick', 'magenta',
                                       'purple'])
    cmap_dbz.set_under('w', alpha=1.0)
    cmap_dbz.set_over('gray', alpha=1.0)

    unit_dbz = "(dBZ)"
    unit_crg = r'(nC m$^{-3}$)'
 
    if vname1 == "QCRG" or vname1 == "CR":
       levs_dbz = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.01,
                             0.01, 0.05, 0.1, 0.2, 0.3, 0.4, ])
       #levs_dbz = np.array([ -2.4, -2.0, -1.6, -1.2, -0.8, -0.4,
       #                       0.4, 0.8, 1.2, 1.6, 2, 2.4])
       cmap_dbz = cmap_rb
       unit_dbz = r'(nC m$^{-3}$)'
       fac = 1.e-6
       fac = 1.0

    elif vname1 == "QHYD":
       levs_dbz = np.array([0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
       unit_dbz = "(g/kg)"
       fac = 1.e3

    elif vname1 == "U" or vname1 == "V" or vname1 == "W":
       #levs_dbz = np.array([ -36, -30, -24, -18, -12, -6,  
       #                       6, 12, 18, 24, 30, 36])
       levs_dbz = np.array([ -48, -40, -32, -24, -16, -8,  
                              8, 16, 24, 32, 40, 48 ])
       cmap_dbz = cmap_rb
       unit_dbz = "(m/s)"



    levs_l = [ levs_dbz, levs_dbz, levs_dbz, 
#               levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg,
               levs_dbz, levs_dbz, levs_dbz, levs_dbz, levs_dbz, levs_dbz,
             ]
#               levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg, levs_rb_qcrg, ]

    cmap_l = [ cmap_dbz, cmap_dbz, cmap_dbz, 
#               cmap_rb, cmap_rb, cmap_rb,
               cmap_dbz, cmap_dbz, cmap_dbz, cmap_dbz, cmap_dbz, cmap_dbz,
               cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb ]
    unit_l = [ unit_dbz, unit_dbz, unit_dbz,
#               unit_crg, unit_crg, unit_crg,
               unit_dbz, unit_dbz, unit_dbz,
               unit_crg, unit_crg, unit_crg ]

    pnum_l = [
              "(a)", "(b)", "(c)",
              "(d)", "(e)", "(f)",
              ]

    tvar = vname2
    if vname1 is "QCRG":
       cmap = cmap_rb
       unit = unit_crg
       tvar = "Total charge density"


    bbox = { 'facecolor':'w', 'alpha':0.95, 'pad':1.5, 'edgecolor':'w' }


    xmin = 120
    xmax = 280
    ymin = 120
    ymax = 320
    zmin = 0.0
    zmax = 15.0

#    xmax -= 40
#    ymax -= 40

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
    cy, cx = np.unravel_index( np.argmax(z_nat[zlev_show,:,:]), ew_nat[0,0,:,:].shape)

    #cx = 76
    #cy = 89

    cy = 110
    cx = 80

    cx = 90
    cy = 89

    print("CX,CY:", cx, cy)


    if typ is not "fcst":
       info = 't={0:.0f}min\nZ={1:}km'.format( ft_sec_a/60.0, INFO["Z"][zlev_show]/1000)
    else:
       info = 't={0:.0f}min\n(FT={1:.0f}min)\nZ={2:}km'.format( ft_sec_a/60, ft_sec/60, INFO["Z"][zlev_show]/1000)
    

    VAR_l = [ 
              evar_exp1[0,zlev_show,:,:]*fac, 
              evar_exp2[0,zlev_show,:,:]*fac, 
              qh_nat[0,zlev_show,:,:]*fac, 
              np.transpose( evar_exp1[0,:,:,cx] )*fac, 
              evar_exp1[0,:,cy,:]*fac, 
              np.transpose( evar_exp2[0,:,:,cx] )*fac, 
              evar_exp2[0,:,cy,:]*fac, 
              np.transpose( qh_nat[0,:,:,cx] )*fac, 
              qh_nat[0,:,cy,:]*fac, 
             ]

    FP_l = [ np.sum( efp_exp1[0,:,:,:], axis=0 ), 
             np.sum( efp_exp2[0,:,:,:], axis=0 ), 
             np.sum( efp_nat[0,:,:,:], axis=0 ),
             np.sum( efp_exp1[0,:,:,:], axis=0 ),
             np.sum( efp_exp2[0,:,:,:], axis=0 ),
             np.sum( efp_nat[0,:,:,:], axis=0 ), 
            ]


    inf = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/" + EXP2 + "/loc.txt"

    #loc_data = np.loadtxt( inf, delimiter=",", dtype='int')
#    loc_data = np.loadtxt( inf, delimiter=",", dtype='float32')
#    print( loc_data[:,0])
#    print( loc_data[:,1])

    for idx, ax in enumerate(ax_l):
   
       if crs_l[idx] == "TZ":
          continue

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

#       if idx == 1 and tlev == 0:
##          ax.scatter( INFO["X"][loc_data[:,0]-1]*0.001, 
##                      INFO["Y"][loc_data[:,1]-1]*0.001,  
#          ax.scatter( loc_data[:,0], 
#                      loc_data[:,1],  
#                      marker='s', s=5, linewidths=0.4,
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
       
       if idx <= 2:
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

       if idx <= 2:
   
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
      
       if idx == 2:
          ax.text( 1.1, 1.2, info,
                   fontsize=9, transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='bottom', )

       #if idx == 6 or idx == 8 or idx == 10:
       #if idx == 5 or idx == 8 or idx == 10:
       if idx == 4 or idx == 6 or idx == 8:
          tvar_ = tvar
          ax.text( 0.5, 1.15, tvar_,
                   fontsize=13, transform=ax.transAxes,
                   horizontalalignment='center',
                   verticalalignment='center', )


    #### score
    tmin_ = 0
    tmax_ = 30
    dt = 5


    ft_l = np.arange( tmin_, tmax_ + dt, dt )

    # pcolor mesh
    Z_h = np.arange(DZ*0.5, DZ*(ZDIM+1), DZ) - DZ*0.5
    ft_lh = np.arange( tmin_, tmax_ + 2*dt, dt ) - dt*0.5

    y2d, x2d = np.meshgrid( Z_h*0.001, ft_lh )

    cmap_rb = plt.cm.get_cmap("RdBu_r")


    ax4 = ax_l[-3]
    ax5 = ax_l[-2]
    ax6 = ax_l[-1]

  
    vname_l = [ vname2, vname3, vname4 ]

    pnum_l = [ "(d)", "(e)", "(f)" ]

    for idx, ax in enumerate( [ ax4, ax5, ax6 ] ):
        
        vname = vname_l[idx]
        pnum = pnum_l[idx]

        if vname == "W":
           unit = "(m/s)"
        elif vname == "QHYD":
           unit = "(g/kg)"
        elif vname == "QCRG":
           unit = r'(nC m$^{-3}$)'

#        rEXP = "2000m_NODA_0306"
        err_t1, err1d_t1, bias_t1, bias1d_t1 = get_dat( INFO, rEXP=NEXP, EXP1=EXP1, vname1=vname, time0=time0 )
        err_t2, err1d_t2, bias_t2, bias1d_t2 = get_dat( INFO, rEXP=NEXP, EXP1=EXP2, vname1=vname, time0=time0 )


        if vname == "QHYD" or vname == "QR" or vname == "QS" or \
           vname == "QG" or vname == "QI" or vname == "QV":
            unit = "(g/kg)"
            dv = 0.02
            dv = 0.01
        elif vname == "QCRG" or vname == "CG" or vname == "CS" or \
            vname == "CR" or vname == "CC" or vname == "CI":
            unit = r'(nC m$^{-3}$)'
            dv = 0.01
            dv = 0.005
            dv = 0.001
        elif vname == "U" or vname == "V" or vname == "W":
            unit = "(m/s)"
            dv = 0.1
#            dv = 0.01
        elif vname == "T":
            unit = "(K)"
            dv = 0.1
    
        vmax = 5*dv
        vmin = -5*dv
        levs = [ -5*dv, -4*dv, -3*dv, -2*dv,-dv, -0.5*dv,
                 0.5*dv, dv, 2*dv, 3*dv, 4*dv, 5*dv]
    

#    ax4.plot( ft_l[:], err_t1, c='k', ls='solid',   label='RMSE NO GLM DA')
#    ax4.plot( ft_l[:], err_t2, c='r', ls='solid',   label='RMSE    GLM DA')
#    ax4.plot( ft_l[:], bias_t1, c='k', ls='dashed', label='Bias   NO GLM DA')
#    ax4.plot( ft_l[:], bias_t2, c='r', ls='dashed', label='Bias      GLM DA')
#
#    ax4.legend()

        norm = BoundaryNorm( levs, ncolors=cmap_rb.N, clip=True )
    

        # 1 is NODA
        # 2 is DA
        print("")
        print( "DIF: {0:} -minus- {1:}".format( EXP2, EXP1 ) )
        print("")
        SHADE = ax.pcolormesh( x2d, y2d, err1d_t2 - err1d_t1,
                               cmap=cmap_rb,
                               norm=norm,
                               vmax=vmax, vmin=vmin, )
    
        pos = ax.get_position()
        cb_h = pos.height #0.01 #pos.height
        cb_w = 0.008
        ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
    
        cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical',
                           extend='both', ticks=levs[::1] )
        cb.ax.tick_params( labelsize=7 )
    
        ax.text(  1.03, -0.02, "improved",
                  fontsize=10, transform=ax.transAxes,
                  color='royalblue',
                  horizontalalignment='left',
                  verticalalignment='top', )

        ax.text(  1.03, 1.02, "degraded",
                  fontsize=10, transform=ax.transAxes,
                  color='firebrick',
                  horizontalalignment='left',
                  verticalalignment='bottom', )


#    ax4.set_xlim( tmin_, tmax_)
        ax.set_xlim( tmin_, tmax_)
        ax.set_ylim( 0, 15.0)

#    ax4.set_xlabel( "Forecast time (min)")
#    ax4.set_ylabel( "RMSE/Bias")
        ax.set_xlabel( "Forecast time (min)")
        ax.set_ylabel( "Height (km)")

#    ax4.text( 0.5, 1.01, vname1 + " RMSE/Bias " + unit_dbz,
#              fontsize=10, transform=ax4.transAxes,
#              horizontalalignment='center',
#              verticalalignment='bottom', )

        ax.text( 0.5, 1.01, vname + " RMSE DIF " + unit,
                  fontsize=10, transform=ax.transAxes,
                  horizontalalignment='center',
                  verticalalignment='bottom', )

#    ax4.text(0.1, 0.95, "(d)",
#             fontsize=10, transform=ax4.transAxes,
#             horizontalalignment='center',
#             verticalalignment='top', 
#             bbox=bbox )

        ax.text(0.1, 0.95, pnum,
                 fontsize=10, transform=ax.transAxes,
                 horizontalalignment='center',
                 verticalalignment='top', 
                 bbox=bbox )
 

#       fig.suptitle( "RMSE/Bias: " + vname1 + " " + unit, fontsize=16 )





#    fig_tit =  tvar
#    fig.suptitle( fig_tit, fontsize=16 )


    odir = 'png/9p_DA_var/{0:}'.format( EXP2  )

    ofig = '9p_{0:}_{1:}_{2:}_fta{3:05}_ft{4:05}_z{5:0=2}_{6:}_{7:}_{8:}.png'.format(typ, EXP1, EXP2, ft_sec_a, ft_sec, zlev_show, vname1, vname2, vname3 )


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
    sys.exit()




###################

DX = 2000.0
DY = 2000.0
TDIM = 13

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
#tmax = 1
#tmin = 1
tmax = 7

typ = "anal"
typ = "fcst"

vname = "QCRG"

tit_l = ["NODA", "DA", "Nature run"]
EXP1 = "2000m_NODA_0306"
EXP2 = "2000m_DA_0306"



EXP1 = "2000m_DA_0306_NOFP"
EXP2 = "2000m_DA_0306_FP_M32_LOC90km"

#EXP1 = "2000m_DA_0306_FP_M32_LOC90km_HT8"

EXP2 = "2000m_DA_0306_FP_M32_LOC90km"
#EXP2 = "2000m_DA_0306_FP_M32_LOC30km"
#EXP2 = "2000m_DA_0306_FP_M32_LOC150km"


EXP1 = "2000m_DA_0601_NOFP"


NEXP = "2000m_NODA_0601"


EXP1 = "2000m_DA_0723_NOFP"
EXP2 = "2000m_DA_0723_FP"

EXP2 = "2000m_DA_0723_FP_M160"
EXP2 = "2000m_DA_0723_FP_NOB"

#EXP2 = "2000m_DA_0723_FP_NOB_30km"

EXP2 = "2000m_DA_0723_FP_NOB_OBERR0.1"

EXP1 = "2000m_DA_0723_NOFP_30min"
EXP2 = "2000m_DA_0723_FP_30min"

#EXP2 = "2000m_DA_0723_FP_30min_NOB"
#EXP2 = "2000m_DA_0723_FP_30min_M64"

#EXP1 = "2000m_DA_0723_FP_30min" # DEBUG
#EXP2 = "2000m_DA_0723_FP_30min_M64" # DEBUG
EXP2 = "2000m_DA_0723_FP_30min_M160"
EXP2 = "2000m_DA_0723_FP_30min_M160_POB"

EXP2 = "2000m_DA_0723_FP_30min_M160_GE3"
EXP2 = "2000m_DA_0723_FP_30min_HT8"

EXP2 = "2000m_DA_0723_FP_30min_LOC30km"
#EXP2 = "2000m_DA_0723_FP_30min"

EXP2 = "2000m_DA_0723_FP_30min_LOC10km_HT16"
#EXP2 = "2000m_DA_0723_FP_30min_LOC10km"

EXP2 = "2000m_DA_0723_FP_30min_LOC30km_X175km_Y183km"

EXP2 = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km"
EXP2 = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km_LOG"

EXP2 = "2000m_DA_0723_FP_30min_LOC90km_LOC2D"

#EXP2 = "2000m_DA_0723_FP_30min_LOC10km_X167km_Y223km"

EXP2 = "2000m_DA_0723_FP_30min_LOC10km"
EXP2 = "2000m_DA_0723_FP_30min_LOC10km_LOG"

EXP2 = "2000m_DA_0723_FP_30min_LOC10km_VLOCW20"

EXP2 = "2000m_DA_0723_FP_30min_LOC10km_VLOC30km"
EXP2 = "2000m_DA_0723_FP_30min_LOC30km"

EXP2 = "2000m_DA_0723_FP_30min_LOC30km_X183km_Y199km"

EXP2 = "2000m_DA_0723_Z20km_FP_30min_LOC30km"

EXP2 = "2000m_DA_0723_FP_30min_LOC30km_COERR0.2"
EXP2 = "2000m_DA_0723_FP_30min_LOC20km"

EXP2 = "2000m_DA_0723_FP_30min_LOC20km_X159km_Y231km"

EXP2 = "2000m_DA_0723_FP_30min_LOC30km_COERR0.2"

EXP2 = "2000m_DA_0723_FP_30min_LOC20km_M240"

EXP2 = "2000m_DA_0723_FP_30min_LOC20km_M240_NOG"

EXP2 = "2000m_DA_0723_FP_30min_LOC20km_M240_NOCLROBS"
NEXP = "2000m_NODA_0723"



tit_l = ["GUESS", "ANAL", "Nature run"]
#tit_l = ["GUESS", "ANAL(HT8)", "Nature run"]


vname1 = "QR"
vname2 = "CR"
#vname1 = "QG"
#vname2 = "CG"
#vname1 = "QS"
#vname2 = "CS"
vname1 = "QHYD"
vname2 = "QCRG"


zlev_min = 6
zlev_max = 28
dz = 4

if typ is not "fcst":
   tmin = 1


#tit_l = ["NO GLM DA\nforecast", "GLM DA\nforecast", "Nature run"]
tit_l = ["NO GLM DA\nanalysis", "GLM DA\nanalysis", "Nature run"]
#tit_l = ["GLMDA forecast\nguess", "GLMDA forecast\nfrom analysis", "Nature run"]
tit_l = ["Forecast from\nGLMDA guess", "Forecast from\nGLMDA analysis", "Nature run"]
tit_l = ["NO GLMDA\nforecast", "GLMDA\nforecast", "Nature run"]
tlev = 3
tlev = 0
tlev = 4
tlev = 2
tlev = 3
#tlev = 0
zlev_show = 22
zlev_show = 16

#zlev_show = 8
#zlev_show = 14
#zlev_show = 18
#zlev_show = 22
zlev_show = 24

zlev_show = 26
zlev_show = 27
#zlev_show = 10
#zlev_show = 8


#for idx, vname1 in enumerate(vname1_l):
#vname2 = vname2_l[idx]
INFO["time0"] = time0

vname2 = "QHYD"
vname1 = "QHYD"
vname3 = "W"
vname4 = "QCRG"

main( INFO, EXP1=EXP1, EXP2=EXP2, NEXP=NEXP, tlev=tlev, typ=typ, tit_l=tit_l, vname1=vname1, vname2=vname2, vname3=vname3, vname4=vname4, zlev_show=zlev_show, time0=time0 )
