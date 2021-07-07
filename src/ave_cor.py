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

from tools_LT import read_evar4d_nc, read_evars, get_ecor, get_eGLM

quick = True
#quick = False

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

def read_ecor( INFO, stime=datetime(2001,1,1,1,0), nvar_l=["tbb"], nvar_ref="QG", tlev=0, zlev_tgt=-1 ):

    ECOR = { "nvar_l":nvar_l }
    AECOR = { "nvar_l":nvar_l }
    CNT = { "nvar_l":nvar_l }

    ctime = stime.strftime('%Y%m%d%H%M%S')
    for nvar in nvar_l:
        if nvar == "tbb" or nvar == "glm" or nvar == "esfc":
           tzlev_tgt = -1
        else:
           tzlev_tgt = zlev_tgt
        ofn = os.path.join( INFO["GTOP"], INFO["EXP"], ctime, "fcst", "ecor_" + nvar + "_" + nvar_ref + "_t" + str(INFO["DT"] * tlev).zfill(5) + "_z" + str(tzlev_tgt).zfill(3) + ".npz")

        print( ofn )

        ECOR[nvar] = np.load( ofn, allow_pickle=True )["ecor"]
        AECOR[nvar] = np.load( ofn, allow_pickle=True )["aecor"]
        CNT[nvar] = np.load( ofn, allow_pickle=True )["num"]

    return( ECOR, AECOR, CNT )

def store_ecor( INFO, ecor, aecor, num=1, stime=datetime(2001,1,1,1,0), nvar="tbb", nvar_ref="QG", tlev=0, zlev_tgt=-1, mem_min=1, fp_acum=1, band=13, CLD=False, qhyd_min=0.001 ):

    if nvar == "tbb" or nvar == "glm" or nvar == "esfc":
       tzlev_tgt = -1
    else:
       tzlev_tgt = zlev_tgt

    if mem_min == 1:
       mem_min_ = ""
    else:
       mem_min_ = "_mem" + str(mem_min).zfill(3)

    if nvar == "tbb":
       band_ = "_B" + str( band ).zfill(2)
    else:
       band_ = ""

    if CLD:
       cld_ = "_cld_qmin{0:.4f}".format( qhyd_min )
    else:
       cld_ = ""

    ctime = stime.strftime('%Y%m%d%H%M%S')
    ofn = os.path.join( INFO["GTOP"], INFO["EXP"], ctime, "fcst", "20210624ecor_" + nvar + "_" + nvar_ref + "_t" + str(INFO["DT"] * tlev).zfill(5) + "_z" + str(tzlev_tgt).zfill(3) + mem_min_ + "_ac" + str( fp_acum ).zfill(2) + band_ + cld_ + ".npz")

    print( ofn )
    np.savez( ofn, ecor=ecor, aecor=aecor, num=num )

def plot_ecor( INFO, tlev=0, vname="QG", member=80, zlev_tgt=10 ):

    nvar_l = ["tbb", "z", "vr", "glm", "fp", "esfc"]

    ECOR, AECOR, CNT = read_ecor( INFO, stime=INFO["time0"], nvar_l=nvar_l, nvar_ref=vname, tlev=tlev, zlev_tgt=zlev_tgt )

    ecor_l = []
    for nvar in nvar_l:
        ecor_l.append( ECOR[nvar]/ CNT[nvar])
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

    xlabel = "X (km)"
    ylabel = "Y (km)"

    for n, var3d in enumerate(nvar_l):

        fig, ((ax)) = plt.subplots(1, 1, figsize=( 7, 7))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.2, hspace=0.2)
 
        rz2d, r1d = calc_rz_crs( var3d=ecor_l[n] )

        r2d, z2d = np.meshgrid( r1d*0.001, z1d*0.001 )

        SHADE = ax.contourf( r2d, z2d, np.transpose(rz2d),
                              cmap=cmap_rb, levels=levs,
                              extend='both' )

        CONT = ax.contour( r2d, z2d, np.transpose(rz2d),
                            levels=levs_c, colors="w", linewidths=2.0,
                            linestyles='solid',
                            )

        ax.clabel( CONT, CONT.levels, inline=True, inline_spacing=-5, fontsize=16, fmt='%3.1f', colors="k" )

        zlev_tgt_ = zlev_tgt
        if var3d != "tbb" and var3d != "glm" and var3d != "esfc":
           ax.plot( 0.0, INFO["Z"][zlev_tgt]*0.001, 
                    linewidth=3.0,
                    marker='s', markersize=15, color='k' )
        else:
           zlev_tgt_ = -1

        pos = ax.get_position()
        cb_h = pos.height #0.01 #pos.height
        cb_w = 0.01
        ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )

        cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical', 
                           ticks=levs, extend='both' )

        ax.set_xlabel( xlabel, fontsize=10 )
        ax.set_ylabel( ylabel, fontsize=10 )

        ax.text( 0.5, 1.05, 'Ensemble-based correlations btw {:1} & {:2}'.format(var3d, vname),
                 fontsize=16, transform=ax.transAxes,
                 horizontalalignment='center',
                 verticalalignment='top', )

        print(n, var3d, vname)

        odir = "png/1p_ave_cor_" + INFO["EXP"]
        ofig = '1p_{:1}_{:2}_z{:0=2}_{:4}'.format( var3d, vname, zlev_tgt_, INFO["EXP"] )

        print( ofig, odir )
 
        if not quick:
           os.makedirs(odir, exist_ok=True)
           plt.savefig(os.path.join(odir,ofig),
                       bbox_inches="tight", pad_inches = 0.1)
           plt.clf()
           plt.close('all')
        else:
           plt.show()
    






def calc_ecor( INFO, tlev=0, vname="QG", member=80, zlev_tgt=10, mem_min=1, fp_acum=1, band=13, CLD=False, qhyd_min=0.001 ):

    if CLD:
       eqhyd2d = np.max( read_evar4d_nc( INFO, vname="QHYD", tlev=tlev, typ="fcst", stime=INFO["time0"] ), axis=1 )
       mqhyd2d = np.mean( eqhyd2d, axis=0 )
       print( eqhyd2d.shape, mqhyd2d.shape )


    # read obs variables & ens variables
    etbb, ez, efp, evar = read_evars( INFO, tlev=tlev, vname=vname, member=member )

    print( "CHK0",etbb.shape, ez.shape, evar.shape )
    _, _, nymax, nxmax = evar.shape

    for dt_ in range( 1, fp_acum ):
       _, _, _, efp_ = read_evars( INFO, tlev=tlev-dt_, vname=vname, member=member )
       efp += efp_


    print( "CHK", vname, evar.shape, "\n" )

    fp_mean = np.mean( efp[1:,:,:,:], axis=0 )
  
#    ew = read_evar4d_nc( INFO, vname="W", tlev=tlev, typ="fcst", stime=INFO["time0"] )



    cvar_mean = np.mean( ez[1:,:,:,:], axis=0 )
       

    # for GLM
    # simply accumulate
    ng = 4
    kernel = np.ones((ng,ng))         # accumulate
    eglm = get_eGLM( efp, kernel )

#    eglm_max2d =  np.max(eglm[1:,:,:], axis=0 ) 
#    idx1, idx2 =  np.where(eglm_max2d[:,:] > 0.0)

    print( eglm.shape, efp.shape )
    eglm_cnt = np.where( eglm > 0.0, 1, 0.0) # 1: on, 0: off
    eglm_cnt2d =  np.sum(eglm_cnt[1:,:,:], axis=0 ) 

    if CLD:
       idx1, idx2 =  np.where( ( eglm_cnt2d[:,:] > mem_min ) & 
                                 ( mqhyd2d[:,:] > qhyd_min ) )
    else:
       idx1, idx2 =  np.where(eglm_cnt2d[:,:] > mem_min )

    jidxs = idx1[(idx1%ng==0) & (idx2%ng==0)]
    iidxs = idx2[(idx1%ng==0) & (idx2%ng==0)]
    
    print( "Number of samples:", len( iidxs ), np.max(eglm_cnt2d), np.min(eglm_cnt2d) )

    #ecor1d_tbb = np.zeros( cvar_mean.shape[0] )

    print("yidx",np.max(jidxs), np.min(jidxs) )
    print("xidx",np.max(iidxs), np.min(iidxs) )

    cnx = 30
    cnx = 20
    cny = cnx

    ecor_tbb = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )
    aecor_tbb = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )

    ecor_z = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )
    aecor_z = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )

    ecor_vr = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )
    aecor_vr = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )

    ecor_glm = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )
    aecor_glm = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )

    ecor_fp = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )
    aecor_fp = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )

    ecor_esfc = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )
    aecor_esfc = np.zeros( (cvar_mean.shape[0], cny*2+1, cnx*2+1 ) )


    print( "NUM:", len(jidxs) )
    for n, cy in enumerate(jidxs):
#        print(n, cy)
        cx = iidxs[n]
        print(cx,cy,n)
        if cx <= cnx or cy <= cny or cx >= ( nxmax - cnx ) or cy >= ( nymax - cny ):
           print( "skip")
           continue

        evar_ref = evar[:,:,cy-cny:cy+cny+1,cx-cnx:cx+cnx+1]
        print( evar_ref.shape )
 
        var1d_tbb = etbb[:,band-7,cy,cx]
        ecor_tbb += get_ecor( var1d_tbb, evar_ref )
        aecor_tbb += np.abs( get_ecor( var1d_tbb, evar_ref ) )

        var1d_z = ez[:,zlev_tgt,cy,cx]
        ecor_z += get_ecor( var1d_z, evar_ref )
        aecor_z += np.abs( get_ecor( var1d_z, evar_ref ) )

#        var1d_vr = evr[:,zlev_tgt,cy,cx]
#        ecor_vr += get_ecor( var1d_vr, evar_ref )
#        aecor_vr += np.abs( get_ecor( var1d_vr, evar_ref ) )

        var1d_glm = eglm[:,cy,cx]
        ecor_glm += get_ecor( var1d_glm, evar_ref )
        aecor_glm += np.abs( get_ecor( var1d_glm, evar_ref ) )

        var1d_fp = efp[:,zlev_tgt,cy,cx]
        ecor_fp += get_ecor( var1d_fp, evar_ref )
        aecor_fp += np.abs( get_ecor( var1d_fp, evar_ref ) )

#        var1d_esfc = np.abs( eetot[:,0,cy,cx] )
#        ecor_esfc += get_ecor( var1d_esfc, evar_ref )
#        aecor_esfc += np.abs( get_ecor( var1d_esfc, evar_ref ) )


#    ecor_tbb = ecor_tbb / len(iidxs)
#    ecor_z = ecor_z / len(iidxs)
#    ecor_vr = ecor_vr / len(iidxs)
#    ecor_glm = ecor_glm / len(iidxs)
#    ecor_fp = ecor_fp / len(iidxs)
#    ecor_esfc = ecor_esfc / len(iidxs)

#    ecor_l = [ ecor_tbb, ecor_z, ecor_vr, ecor_glm, ecor_fp, ecor_esfc]
#    aecor_l = [ aecor_tbb, aecor_z, aecor_vr, aecor_glm, aecor_fp, aecor_esfc]
#    nvar_l = [ "tbb", "z", "vr", "glm", "fp", "esfc" ]

    ecor_l = [ ecor_tbb, ecor_z, ecor_glm, ecor_fp, ]
    aecor_l = [ aecor_tbb, aecor_z, aecor_glm, aecor_fp,]
    nvar_l = [ "tbb", "z", "glm", "fp", ]

    ecor_l = [ ecor_tbb, ecor_z, ecor_glm, ecor_fp, ]
    aecor_l = [ aecor_tbb,  aecor_z, aecor_glm, aecor_fp,]
    nvar_l = [ "tbb",  "z", "glm", "fp", ]

    for n, ecor in enumerate(ecor_l):

        store_ecor( INFO, ecor_l[n], aecor_l[n], num=len(iidxs), stime=INFO["time0"], nvar=nvar_l[n], nvar_ref=vname, tlev=tlev, zlev_tgt=zlev_tgt, mem_min=mem_min, fp_acum=fp_acum, band=band,
                    CLD=CLD )


       

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
tmin = 1
#tmin = 2 # 600
#tmin = 3
tmax = tmin + 1
vname = "QG"
#vname = "CS"
vname = "QS"

#vname_l = [ "QHYD", "QCRG", "QG", "QS", "QC", "QR", "QI", "QV", "CR", "CI", "CC",
vname_l = [ "QCRG", "QG", "QS", "QC", "QR", "QI", "QV", "CR", "CI", "CC",
            "CG", "CS", "W", "T", "U", "V"]



vname_l = [ "W", "QV", "T", "QHYD"]
vname_l = [ "QG", "QS", "QR", "QI", "QC"]

vname_l = [ "W", "QV", "T", "QHYD", "QG", "QS", "QR", "QI", "QC" ]
vname_l = [ "QHYD", ]

zlev_tgt = 13
zlev_tgt = 16
zlev_tgt = 20
zlev_tgt = 10
#zlev_tgt = 5

mem_min = 1
mem_min = 32
mem_min = 64
mem_min = 160

mem_min = 32
#mem_min = 160
#mem_min = 128
#mem_min = 1

band = 9
#band = 8
#band = 10
#band = 13

CLD = True
qhyd_min = 0.001
qhyd_min = 0.0001

tmin = 6
tmax = tmin + 1
fp_acum = 6
#fp_acum = 1

if fp_acum > tmin:
   fp_acum = tmin

for vname in vname_l:
    for tlev in range( tmin, tmax, fp_acum ):
        calc_ecor( INFO, tlev=tlev, vname=vname, member=320, zlev_tgt=zlev_tgt, mem_min=mem_min, fp_acum=fp_acum, band=band, CLD=CLD, qhyd_min=qhyd_min )
        #plot_ecor( INFO, tlev=tlev, vname=vname, member=320, zlev_tgt=zlev_tgt )

