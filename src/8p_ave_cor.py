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


quick = True
quick = False

def calc_rz_crs( var3d=np.array([]) ):

    nz = var3d.shape[0]
    nx = (var3d.shape[2] - 1) / 2
    ny = (var3d.shape[1] - 1) / 2

    DX = 2000
    x1d = np.arange(-nx*DX, (nx+1)*DX, DX)
    y1d = np.arange(-nx*DX, (ny+1)*DX, DX)
    

    x2d, y2d = np.meshgrid( y1d, x1d )

    dist2d = np.sqrt( np.square(x2d) + np.square(y2d) )

    DR = 2000.0
    r1d = np.arange(0, nx*DX+DR, DR )
    nr = r1d.shape[0]

    rz2d = np.zeros( (nr, nz))
    rz2d_cnt = np.zeros( (nr, nz))

    ridxs = np.rint( dist2d/DR ).astype(np.int32)

    for z in range(nz):
        for r in range(nr):
            var2d = var3d[z,:,:]
            rz2d[r,z] += np.sum( var2d[ ridxs == r ] )
            rz2d_cnt[r,z] += len( var2d[ ridxs == r ] )

    rz2d = rz2d / rz2d_cnt

    return( rz2d, r1d )

def read_ecor( INFO, stime=datetime(2001,1,1,1,0), nvar_l=["tbb"], nvar_ref="QG", tlev=0, zlev_tgt=-1, mem_min=1, fp_acum=1 ):

    ECOR = { "nvar_l":nvar_l }
    AECOR = { "nvar_l":nvar_l }
    CNT = { "nvar_l":nvar_l }

    if mem_min == 1:
       mem_min_ = ""
    else:
       mem_min_ = "_mem" + str(mem_min).zfill(3)

    ctime = stime.strftime('%Y%m%d%H%M%S')
    for nvar in nvar_l:



        if nvar == "tbb" or nvar == "glm" or nvar == "esfc":
           tzlev_tgt = -1
        else:
           tzlev_tgt = zlev_tgt
##        ofn = os.path.join( INFO["GTOP"], INFO["EXP"], ctime, "fcst", "ecor_" + nvar + "_" + nvar_ref + "_t" + str(INFO["DT"] * tlev).zfill(5) + "_z" + str(tzlev_tgt).zfill(3) + ".npz")
#        ofn = os.path.join( INFO["GTOP"], INFO["EXP"], ctime, "fcst", "ecor_" + nvar + "_" + nvar_ref + "_t" + str(INFO["DT"] * tlev).zfill(5) + "_z" + str(tzlev_tgt).zfill(3) + mem_min_ + ".npz")
        ofn = os.path.join( INFO["GTOP"], INFO["EXP"], ctime, "fcst", "ecor_" + nvar + "_" + nvar_ref + "_t" + str(INFO["DT"] * tlev).zfill(5) + "_z" + str(tzlev_tgt).zfill(3) + mem_min_ + "_ac" + str( fp_acum ).zfill(2) + ".npz")

        print( ofn )

        ECOR[nvar] = np.load( ofn, allow_pickle=True )["ecor"]
        AECOR[nvar] = np.load( ofn, allow_pickle=True )["aecor"]
        CNT[nvar] = np.load( ofn, allow_pickle=True )["num"]

    return( ECOR, AECOR, CNT )

def plot_ecor( INFO, nvar_l=[],tlev=0, vname="QG", member=80, zlev_tgt=10, mem_min=1, fp_acum=1, 
               vname_l=["QG", "W", "T"] ):
    
    tvar_l = []
    info_l = []
    ecor_l = []
    for i, vname in enumerate( vname_l ):
        ECOR, AECOR, CNT = read_ecor( INFO, stime=INFO["time0"], nvar_l=nvar_l, nvar_ref=vname, tlev=tlev, zlev_tgt=zlev_tgt, mem_min=mem_min, fp_acum=fp_acum )

        ecor_l.append( ECOR["tbb"]/ CNT["tbb"])
        info_l.append( "tbb" )
        tvar_l.append( vname )

    for i, vname in enumerate( vname_l ):
        ECOR, AECOR, CNT = read_ecor( INFO, stime=INFO["time0"], nvar_l=nvar_l, nvar_ref=vname, tlev=tlev, zlev_tgt=zlev_tgt, mem_min=mem_min, fp_acum=fp_acum )
        ecor_l.append( ECOR["glm"]/ CNT["glm"])
        info_l.append( "glm" )
        tvar_l.append( vname )
    print( info_l )

        #ecor_l.append( AECOR[nvar]/ CNT[nvar])

    cmap_rb = plt.cm.get_cmap("RdBu_r")
    cmap_rb.set_over('gray', alpha=1.0)
    cmap_rb.set_under('gray', alpha=1.0)

    #levs = np.arange(-0.3,0.35,0.05)
    #levs_c = np.arange(-0.9,1.0,0.1)
    levs = [ -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 
             0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    levs_c = [ -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    z1d = INFO["Z"]

    xlabel = "Horizontal distance (km)"
    ylabel = "Height (km)"


    dv = 8
    dv2 = 2  
    dh = 6      
    dh2 = 2  

    v_tot = dv + dv2 + dv
    h_tot = dh*4 + dh2*3 

    hsize = 11.0
    vsize = hsize * v_tot / h_tot
    fig = plt.figure( figsize=(hsize, vsize) )

    gs = gridspec.GridSpec( 3, 7, 
                            height_ratios=( dv, dv2, dv ), 
                            width_ratios=( dh, dh2, dh, dh2, dh, dh2, dh ) )
    axs = [ 
             plt.subplot(gs[0, 0]), 
             plt.subplot(gs[0, 2]), 
             plt.subplot(gs[0, 4]), 
             plt.subplot(gs[0, 6]), 

             plt.subplot(gs[2, 0]), 
             plt.subplot(gs[2, 2]), 
             plt.subplot(gs[2, 4]), 
             plt.subplot(gs[2, 6]), 
            ]

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax4 = axs[3]
    ax5 = axs[4]
    ax6 = axs[5]
    ax7 = axs[6]
    ax8 = axs[7]

    ax_l = [ ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 ]

    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.92, top=0.95,
                        wspace=0.0, hspace=0.0)

#    fig, (( ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots( 2, 4, figsize=( 10, 6))
##                        wspace=0.3, hspace=0.4)
 
    pnum_l = [ "(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)" ]
#    ax_l = [ ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 ]
    bbox = { 'facecolor':'w', 'alpha':0.95, 'pad':1.5, 'edgecolor':'w' }

    for n, ax in enumerate( ax_l ):
 
        rz2d, r1d = calc_rz_crs( var3d=ecor_l[n] )

        r2d, z2d = np.meshgrid( r1d*0.001, z1d*0.001 )

        SHADE = ax.contourf( r2d, z2d, np.transpose(rz2d),
                              cmap=cmap_rb, levels=levs,
                              extend='both' )

        CONT = ax.contour( r2d, z2d, np.transpose(rz2d),
                            levels=levs_c, colors="w", linewidths=2.0,
                            linestyles='solid',
                            )

        ax.clabel( CONT, CONT.levels, inline=True, inline_spacing=0, 
                   fontsize=8, fmt='%3.1f', colors="k" )

        ax.tick_params(axis='both', which='minor', labelsize=8 )
        ax.tick_params(axis='both', which='major', labelsize=8 )

        ymin = 0.0
        ymax = 17.0
        ax.set_ylim( ymin, ymax )

        zlev_tgt_ = zlev_tgt
#        if var3d != "tbb" and var3d != "glm" and var3d != "esfc":
#           ax.plot( 0.0, INFO["Z"][zlev_tgt]*0.001, 
#                    linewidth=3.0,
#                    marker='s', markersize=15, color='k' )
#        else:
#           zlev_tgt_ = -1

        ax.set_xlabel( xlabel, fontsize=9 )
        ax.set_ylabel( ylabel, fontsize=9 )

        ax.text( 0.08, 0.95, pnum_l[n],
                 fontsize=9, transform=ax.transAxes,
                 horizontalalignment='center',
                 verticalalignment='top', 
                 bbox=bbox )

        if info_l[n] == "esfc":
           var3d_ = "Surface Ez"
        elif info_l[n] == "tbb":
           var3d_ = r'IR (10.4$\mu$m)'
        else:
           var3d_ = str.upper( info_l[n] )
        ax.text( 0.5, 0.95, '{:1} & {:2}'.format( var3d_, tvar_l[n] ),
                 fontsize=10, transform=ax.transAxes,
                 horizontalalignment='center',
                 verticalalignment='top', 
                 bbox=bbox )


    pos = ax8.get_position()
    cb_h = pos.height*1.5 #0.01 #pos.height
    cb_w = 0.01
    ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0+pos.height*0.5, 
                  cb_w, cb_h] )

    cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical', 
                       ticks=levs, extend='both' )
    cb.ax.tick_params( labelsize=8 )

    fig.suptitle( "Averaged ensemble-based correlations", fontsize=12 )

    odir = "png/8p_ave_cor_" + INFO["EXP"]
    ofig = '8p_{:}_{:}_z{:0=2}_{:}_lm{:0=3}_ac{:0=2}'.format( info_l[n], vname, zlev_tgt_, INFO["EXP"], mem_min, fp_acum )

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
TDIM = 7

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
EXP = "2000m_DA_0306"
EXP = "2000m_DA_0601"
EXP = "2000m_DA_0723"

time0 = datetime( 2001, 1, 1, 1, 20, 0 ) 
time0 = datetime( 2001, 1, 1, 1, 30, 0 ) 

TOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/" + EXP
GTOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT" 
TYPE = "fcst"
time00 = datetime( 2001, 1, 1, 0, 0, 0 )

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "TOP":TOP, "GTOP":GTOP,
        "ZDIM":ZDIM, "Z":Z, "DT":DT,
        "TYPE":TYPE, "EXP":EXP,
        "time0": time0, "time00": time00  }

tmax = 13
vname = "QG"
#vname = "CS"
vname = "QS"

vname_l = [ "QR", "QI", "QV", "CR", "CI", 
            "QS", "QG", "CG", "CS", "QHYD", "QCRG", "W"]


vname_l = [ "CG", "QHYD", "QCRG","W",
#            "QS", "CS", "QG", "CG", "U", "V",
            ]

#vname_l = [ "QS", "CS"]
vname_l = [ "QG", ]
#vname_l = [ "CG", ]#"CG",]

vname_l = [ "QHYD", "W"]
vname_l = [ "QV", "QC", "QR", "QI", "QS", "QG", "QHYD",
            "CC", "CR", "CI", "CS", "CG", "QCRG",
            "T", "W", "U", "V",
            ]
vname_l = [ "QHYD", ]
vname_l = [ "W", ]
#vname_l = [ "CG", ]

zlev_tgt = 13
zlev_tgt = 16
zlev_tgt = 20
zlev_tgt = 10
#zlev_tgt = 5

mem_min = 1
mem_min = 32
#mem_min = 64
#mem_min = 96
#mem_min = 160

fp_acum = 6
#fp_acum = 1

#tmin = 2 # 600min
#tmin = 3
tmin = 6
tmax = tmin + 1

#nvar_l = ["tbb", "z", "vr", "glm", "fp", "esfc"]
nvar_l = ["tbb", "glm", "esfc"]
#nvar_l = ["vr", "glm", "z"]

nvar_l = ["tbb", "glm", ]

vname_l=["QHYD", "W", "V", "T"]

for tlev in range( tmin, tmax ):
    plot_ecor( INFO, nvar_l=nvar_l, tlev=tlev, vname=vname, member=320, zlev_tgt=zlev_tgt, mem_min=mem_min, fp_acum=fp_acum, 
       vname_l=vname_l )

