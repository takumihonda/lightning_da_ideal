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

quick = True
#quick = False

def read_vars( INFO, tlev=0, acm_fp=1 ):

    # Read variables
    fn_Him8 = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], INFO["MEM"], 
                            "Him8_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".nc") 
    nc = Dataset(fn_Him8, 'r', format='NETCDF4')
    tbb = nc.variables["tbb"][tlev,:,:,:]
    nc.close()

    fn_radar = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], INFO["MEM"], 
                       "radar_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".nc") 
    nc = Dataset(fn_radar, 'r', format='NETCDF4')
    z = nc.variables["z"][tlev,:,:,:]
    vr = nc.variables["vr"][tlev,:,:,:]
    nc.close()

    tlev_min = tlev - acm_fp + 1
    for tlev_ in range( tlev_min, tlev+1 ):
        fn_FP = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], 
                              "FP_ens_t" + str( tlev_ * INFO["DT"] ).zfill(5) + ".nc") 
        nc = Dataset( fn_FP, 'r', format='NETCDF4' )
        if tlev_ == tlev_min:
           fp = nc.variables["FP"][0,:,:,:]
        else:
           fp += nc.variables["FP"][0,:,:,:]
        nc.close()

#    for vname in ["EX", "EY", "EZ", "W"]:
    for vname in [ "EZ", "W"]:
        fn_E = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], 
                             vname + "_ens_t" + str( tlev * INFO["DT"] ).zfill(5) + ".nc") 
        nc = Dataset( fn_E, 'r', format='NETCDF4' )
        #if vname == "EX":
        #   e_tot = np.square( nc.variables["EX"][0,:,:,:] )
        #elif vname is "EY" or vname is "EZ":
        if vname == "EZ":
           e_tot = nc.variables[vname][0,:,:,:] 
        elif vname == "W":
           w = nc.variables[vname][0,:,:,:]
        nc.close()
#    e_tot = np.sqrt( e_tot )

    return( tbb, z, vr, fp, e_tot, w )

def main( INFO, tlev=0, acm_fp=1 ):

    # read variables
    tbb, z, vr, fp, e_tot, w = read_vars( INFO, tlev=tlev, acm_fp=acm_fp )

    ctime = INFO["time0"] + timedelta( seconds = int( tlev ) * INFO["DT"] )
    ft_sec =  int( (ctime - INFO["time00"] ).total_seconds() )

#    fig, ( (ax1,ax2,ax3,ax4), (ax5,ax6,ax7,ax8) ) = \
#    plt.subplots(2, 4, figsize=(14,7.2))

    fig = plt.figure( figsize=(9.5, 9.5) )
    gs = gridspec.GridSpec(85, 100) # v:h

    pdh = 5
    dv = 30
    pdv = 5

    dv_t = 10
    dh = 30
    dh_r = 10
    hmin = 0
    vmax = 0

    hmin1 = 0
    vmax1 = vmax + dv_t + pdv
    vmax1_t = vmax 
    hmin1_r = hmin1 + dh + pdh
    ax1 = plt.subplot(   gs[vmax1:vmax1+dv,hmin1:hmin1+dh] )
    ax1_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,    hmin1:hmin1+dh] )
    ax1_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin1_r:hmin1_r+dh_r] )
    print( "ax1", vmax1,vmax1+dv,hmin1,hmin1+dh)

    hmin2 = hmin1_r + dh_r + 2*pdh
    hmin2_r = hmin2 + dh + pdh
    ax2 = plt.subplot(   gs[vmax1:vmax1+dv,hmin2:hmin2+dh] )
    ax2_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin2:hmin2+dh] )
    ax2_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin2_r:hmin2_r+dh_r] )

    hmin3 = 0
    vmax3 = vmax + dv_t + dv + 3*pdv
    ax3 = plt.subplot(   gs[vmax3:vmax3+dv,hmin3:hmin3+dh] )

    hmin4 = hmin + dh + pdh 
    ax4 = plt.subplot(   gs[vmax3:vmax3+dv,hmin4:hmin4+dh] )

    hmin5 = hmin + ( dh + pdh )*2
    ax5 = plt.subplot(   gs[vmax3:vmax3+dv,hmin5:hmin5+dh] )
    print( "ax5", vmax3,vmax3+dv,hmin5,hmin5+dh)


    fig.subplots_adjust( left = 0.05, right=0.98, top=0.94, bottom=0.1 )
    #fig.subplots_adjust( left=0.0, bottom=0.0, right=0.99, top=0.99,
    #                     wspace=0.0, hspace=0.0 )


    ax_l = [ ax1, ax2, ax3, ax4, ax5, 
             ax1_r, ax1_t, ax2_r, ax2_t ] #,ax2,ax3,ax4,ax5,ax6, ] #, ax5,ax6,ax7,ax8 ]

    colors1 = plt.cm.jet_r(np.linspace(0, 1, 128))
    colors2 = plt.cm.binary(np.linspace(0., 1, 128)) # w/k
    colors = np.vstack((colors1, colors2))
    cmap_tbb = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    levs_tbb = np.arange(200,304,4)


    #cmap_e = cmap_jet = plt.cm.get_cmap( "hot_r" )
    #levs_e = np.arange( 0,100,4 )
    cmap_e = plt.cm.get_cmap("RdBu_r")
    levs_e = np.arange( -80, 88, 8 )
    cmap_e.set_under('gray', alpha=1.0)
    cmap_e.set_over('gray', alpha=1.0)

    cmap_fp = cmap_jet = plt.cm.get_cmap( "hot_r" )
    #levs_fp = np.arange( 0, 3.5, 0.5 )
    levs_fp = np.arange( 0, 5.0, 1.0 ) 
    cmap_fp.set_under('w', alpha=1.0)

    levs_dbz= np.array([ 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    cmap_dbz = mcolors.ListedColormap(['cyan','dodgerblue', 
                                       'lime', 'limegreen','yellow',
                                       'orange', 'red', 'firebrick', 'magenta',
                                       'purple'])
    cmap_dbz.set_under('w', alpha=1.0)
    cmap_dbz.set_over('gray', alpha=1.0)


    zlev = 10
    print( INFO["Z"][zlev])

    # for GLM
    # simply accumulate
    ng = 4
    #ng = 2
    kernel = np.ones((ng,ng))         # accumulate
    #kernel = kernel / np.size(kernel) # average
    from scipy import ndimage

    bbox = { 'facecolor':'w', 'alpha':0.95, 'pad':1.5, 'edgecolor':'w' }

    fp_note = ""
    if acm_fp > 1:
       fp_note = " {0:.0f}-min accumulated\n".format( INFO["DT"] * acm_fp / 60.0 )

    crs_l = [
              "XY_s",
              "XY_s",
              "XY",
              "XY",
              "XY_skip",
              "YZ",
              "XZ",
              "YZ",
              "XZ",
            ]

    tit_l = [ 
              "Radar",
              fp_note + "BOLT (3D flash)", 
              r'IR (10.4$\mu$m)', 
              "Surface Ez", 
              fp_note + "GLM (2D flash)", 
              "",
              "",
              "",
              "",
              "Pseudo BOLT", "", "" 
            ]

    pnum_l = [ "(a)", 
               "(b)",
               "(c)",
               "(d)",
               "(e)",
               ]

    unit_l = [ 
               '(dBZ)', 
               '(flash/' + str( int( INFO["DT"]*acm_fp/60.0 ) ) + r'min)',
               '(K)', 
               '(kV/m)', # E
               '(flash/' + str( int( INFO["DT"]*acm_fp/60.0 ) ) + r'min)', # GLM
               '(dBZ)', 
               '(dBZ)', 
               '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r'min)',
               '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r'min)',
              ]

    res = np.unravel_index( np.argmax(w[:,:,:]), w[:,:,:].shape )
    max_yx = [ res[1], res[2] ]
#    max_yx = np.unravel_index( np.argmax(w[zlev,:,:]), w[zlev,:,:].shape )

    VAR_l = [ 
              z[zlev,:,:], 
              np.sum(fp[:,:,:], axis=0 ), # BOLT
              tbb[13-7,:,:], 
              e_tot[0,:,:]*0.001,
              ndimage.convolve( np.sum( fp[:,:,:], axis=0 ), kernel, mode='reflect' ),  # GLM
              np.transpose( z[:,:,max_yx[1]] ), # radar YZ
              z[:,max_yx[0],:], # radar XZ
              np.transpose( np.sum(fp[:,:,:], axis=2 ) ), # BOLT YZ
              np.sum(fp[:,:,:], axis=1 ), #  BOLT XZ
            ]
 
    levels_l = [ 
                 levs_dbz, 
                 levs_fp, 
                 levs_tbb, 
                 levs_e,   # E
                 levs_fp,  # GLM
                 levs_dbz,
                 levs_dbz,
                 levs_fp, 
                 levs_fp ]
    cmap_l = [ 
               cmap_dbz,
               cmap_fp, 
               cmap_tbb, 
               cmap_e,   # E
               cmap_fp,  # GLM
               cmap_dbz, 
               cmap_dbz, 
               cmap_fp, 
               cmap_fp, 
               ]

#    xmin_l = 120 - 10
#    xmax_l = 300 - 10
#    ymin_l = 140 - 10
#    ymax_l = 320 - 10 

    xmin_l = 120 - 20
    xmax_l = 300 - 20
    ymin_l = 140 - 20
    ymax_l = 320 - 20 

    xmin_s = xmin_l + 20 
    xmax_s = xmax_l - 40 
    ymin_s = ymin_l + 20 
    ymax_s = ymax_l - 40 


    ctime = ( INFO["time0"] + timedelta(seconds=INFO["DT"] * tlev ) ).strftime('%H:%M:%S')

    for idx, ax in enumerate(ax_l):
       print(idx,tit_l[idx])

       xlabel = "X (km)"
       ylabel = "Y (km)"

       if crs_l[idx] is "XY":
          xmin = xmin_l
          xmax = xmax_l
          ymin = xmin_l
          ymax = ymax_l
          xdgrid = 20
          ydgrid = 20

       elif crs_l[idx] is "XY_s" or "XY_skip":
          xmin = xmin_s
          xmax = xmax_s
          ymin = ymin_s
          ymax = ymax_s
          xdgrid = 10
          ydgrid = 10

          if crs_l[idx] is "XY_skip":
             xmin = xmin_l
             xmax = xmax_l
             ymin = xmin_l
             ymax = ymax_l
             xdgrid = 20
             ydgrid = 20
   

       if crs_l[idx] is "XY_skip":
          nskip = ng

          kernel1d = np.ones(ng) / ng
          xaxis = np.convolve( INFO["X"]*0.001, kernel1d, mode='same' )[nskip::nskip]
          yaxis = np.convolve( INFO["Y"]*0.001, kernel1d, mode='same' )[nskip::nskip]
       else:
          nskip = 1
          xaxis = INFO["X"][nskip::nskip] * 0.001
          yaxis = INFO["Y"][nskip::nskip] * 0.001

       imin = nskip
       jmin = nskip

       if crs_l[idx] is "YZ":
          xmin = 0.0
          xmax = 15.0
          ymin = ymin_s
          ymax = ymax_s
          nskip = 1
          imin = 0
          jmin = 0

          xdgrid = 2
          ydgrid = 10

          yaxis = INFO["Z"][:] * 0.001
          xaxis = INFO["Y"][jmin::nskip] * 0.001
          xlabel = "Z (km)"

          if ax is ax1_r or ax is ax2_r:
             ax.vlines( x=INFO["Z"][zlev]*0.001, ymin=ymin, ymax=ymax,
                        colors="k",linestyles='dotted',linewidths=1.0 )

       elif crs_l[idx] is "XZ":
          ymin = 0.0
          ymax = 15.0
          xmin = xmin_s
          xmax = xmax_s

          nskip = 1
          imin = 0
          jmin = 0

          ydgrid = 2
          xdgrid = 10

          xaxis = INFO["Z"][:] * 0.001
          yaxis = INFO["X"][jmin::nskip] * 0.001

          ylabel = "Z (km)"

          if ax is ax1_t or ax is ax2_t:
             ax.hlines( y=INFO["Z"][zlev]*0.001, xmin=xmin, xmax=xmax,
                        colors="k",linestyles='dotted',linewidths=1.0 )

       #if idx == 0 or idx == 4:
       if ax is ax1 or ax is ax2 or ax is ax3 or ax is ax4 or ax is ax5:
          ax.vlines( x=INFO["X"][max_yx[1]]*0.001, ymin=ymin, ymax=ymax,
                     colors="k",linestyles='dotted',linewidths=1.0 )
          ax.hlines( y=INFO["Y"][max_yx[0]]*0.001, xmin=xmin, xmax=xmax,
                     colors="k",linestyles='dotted',linewidths=1.0 )

       ax.set_xlabel( xlabel, fontsize=6 )
       ax.set_ylabel( ylabel, fontsize=6 )

       x2d, y2d = np.meshgrid( yaxis, xaxis )

       print( "CHECK", idx, VAR_l[idx].shape, np.max(VAR_l[idx]), np.min(VAR_l[idx]) )
       #print( x2d.shape, y2d.shape)
       #print("" )

       norm = BoundaryNorm( levels_l[idx], ncolors=cmap_l[idx].N, clip=True )

       if ax is ax2 or ax is ax5 or ax is ax2_t or ax is ax2_r:
          SHADE = ax.pcolormesh(x2d, y2d,
                                VAR_l[idx][jmin::nskip,imin::nskip],
                                vmin=np.min(levels_l[idx]),
                                vmax=np.max(levels_l[idx]),
                                cmap=cmap_l[idx], 
                                norm=norm,
                                #extend='both',
                                ) 
       else:
          SHADE = ax.contourf(x2d, y2d,
                              VAR_l[idx][jmin::nskip,imin::nskip],
                              levels=levels_l[idx],
                              cmap=cmap_l[idx], 
                              norm=norm,
                              extend='both',
                             )

       ax.set_xlim( xmin, xmax )
       ax.set_ylim( ymin, ymax )
       ax.xaxis.set_ticks( np.arange(xmin, xmax, xdgrid) )
       ax.yaxis.set_ticks( np.arange(ymin, ymax, ydgrid) )
       ax.tick_params(axis='both', which='minor', labelsize=6 )
       ax.tick_params(axis='both', which='major', labelsize=6 )


#          ax.set_ylabel( ylabel, fontsize=6 )
#
       ax.grid( axis='both', ls='dashed', lw=0.2 )

       tskip = 1
       #if ax is ax2 or ax is ax3 or ax is ax4 or ax is ax5:
       if ax is ax3 or ax is ax4:
          tskip = 2


       if crs_l[idx] != "XZ" and crs_l[idx] != "YZ":
          pos = ax.get_position()
          cb_h = 0.01 #pos.height
          cb_w = pos.width * 1.0
          ax_cb = fig.add_axes( [pos.x0+0.0, pos.y0-0.055, cb_w, cb_h] )
          cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'horizontal', 
                             ticks=levels_l[idx][::tskip], extend='max' )
          cb.ax.tick_params( labelsize=6 )
          ax.text( 1.0, -0.08, unit_l[idx],
                   fontsize=6, transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top', )
       
   
       #if crs_l[idx] is "XY":
       if crs_l[idx] is "XY" or crs_l[idx] is "XY_skip":
          rect = patches.Rectangle( (xmin_s,ymin_s), 
                                    xmax_s-xmin_s, ymax_s-ymin_s, 
                                    lw=1, edgecolor='r',facecolor='none' )
          ax.add_patch(rect)

#       if idx == 5:
#          ax.tick_params( labelbottom=False ) 
#       if idx == 4:
#          ax.tick_params( labelleft=False ) 

       ax.text(0.5, 0.95, tit_l[idx],
               fontsize=10, transform=ax.transAxes,
               horizontalalignment='center',
               verticalalignment='top', 
               bbox=bbox )
       
       if idx <= 4:
          ax.text(0.1, 0.95, pnum_l[idx],
                  fontsize=10, transform=ax.transAxes,
                  horizontalalignment='center',
                  verticalalignment='top', 
                  bbox=bbox )
       

       if idx == 0:
          fig.text(0.99, 0.96, "t = {0:.0f} min".format( ft_sec / 60.0 ),
                  fontsize=11, #transform=ax.transAxes,
                  horizontalalignment='right',
                  verticalalignment='center')

#       if idx == 2:
#          ax.set_xticks(np.arange(0,300,2), minor=False)

    fig.suptitle( "Nature run", fontsize=18 )


    odir = "png/9p_obs_" + INFO["EXP"]
    ofig =  "9p_nature_obs_t{0:0=5}_acm_fp{1:0=2}".format( ft_sec, acm_fp )

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
#XDIM = 192
#YDIM = 192
#ZDIM = 40
XDIM = 176
YDIM = 176
TDIM = 13
ZDIM = 45

DZ = 500.0
DT = 300

X = np.arange( DX*0.5, DX*XDIM, DX )
Y = np.arange( DY*0.5, DY*YDIM, DY )
XH = np.arange( 0.0, DX*(XDIM+1), DX )
YH = np.arange( 0.0, DY*(YDIM+1), DY )
T = np.arange( 0, DT*TDIM, DT )
BAND = np.arange( 7, 17, 1 )

Z = np.arange( DZ*0.5, DZ*ZDIM, DZ )
ZH = np.arange( 0.0, DZ*(ZDIM+1), DZ )

EXP = "2000m_NODA_0306"
#EXP = "2000m_NODA_0601"
EXP = "2000m_NODA_0723"
TOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/" + EXP
TYPE = "fcst"
MEM = "mean"
time0 = datetime( 2001, 1, 1, 1, 0, 0 )
time00 = datetime( 2001, 1, 1, 0, 0, 0 )

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "TOP":TOP,
        "XH":XH, "YH":YH, "ZH":ZH,
        "ZDIM":ZDIM, "Z":Z, "DT":DT,
        "TYPE":TYPE, "MEM":MEM, "EXP":EXP,
        "time0": time0, "time00": time00  }

tlev = 3

ts = 1
ts = 6
te = 13

#ts = 6

acm_fp = 1
acm_fp = 6

for tlev in range( ts, te ):
    main( INFO, tlev=tlev, acm_fp=acm_fp )
