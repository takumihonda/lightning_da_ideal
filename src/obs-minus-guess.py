import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from tools_LT import read_evar4d_nc, get_eGLM, get_GLM

DEBUG = True
DEBUG = False

quick = True
#quick = False

LOG = False
#LOG = True
log_c = 0.1

HIST = True

# for GLM
# simply accumulate
ng = 4
kernel = np.ones((ng,ng))         # accumulate

def main( INFO, tlev=0, fp_acum=1 ):
    print("")

    fn = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), "fcst", 
               "mean_fp_acum{0:0=2}.nc".format( fp_acum) )

    try: 
       nc = Dataset( fn, 'r', format='NETCDF4')
       fp = nc.variables["fp"][:,:]
       fp_mem = nc.variables["fp_mem"][:,:]
       efp = nc.variables["efp"][:]
       nc.close()
    except: 
       print( "Failed to get")
       print( fn )
       sys.exit()


    # nature run
    INFO["EXP"] = "2000m_NODA_0723"
    INFO["time0"] = datetime( 2001, 1, 1, 1, 0, 0 )
    tlev = 12

    ctime = INFO["time0"] + timedelta( seconds=INFO["DT"]*tlev )

    fn_nat = os.path.join( INFO["GTOP"], INFO["EXP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), "fcst", 
                  "fp_acum{0:0=2}.nc".format( fp_acum) )
    print( fn_nat )

    try: 
       nc = Dataset( fn_nat, 'r', format='NETCDF4')
       fp_nat = nc.variables["fp"][:,:]
       nc.close()
    except: 
       print( "Failed to get natunre run")
       print( fn_nat )
       sys.exit()

    if DEBUG:
       # debug
       fn_Him8 = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/2000m_NODA_0723/20010101010000/fcst/mean/radar_20010101010000_mean.nc"
       nc = Dataset( fn_Him8, 'r', format='NETCDF4')
       Him8_nat = nc.variables["z"][6,10,:,:]
       nc.close()

    x2d, y2d = np.meshgrid( INFO["Y"]*0.001, INFO["X"]*0.001 )



    if HIST:
       #print( "all:",fp_mem[ fp_mem > 300 ] )

       cmap = plt.cm.get_cmap("RdBu_r")
       cmap.set_over('gray', alpha=1.0)
       cmap.set_under('gray', alpha=1.0)


       levs = np.arange( -4, 4.5, 0.5 ) 

       ng2 = int( ng / 2 )

       eglm_ = efp[:,ng2::ng,ng2::ng]
       fp_mem_ = fp_mem[ng2::ng,ng2::ng]
       fp_nat_ = fp_nat[ng2::ng,ng2::ng]        
 
       if LOG:
          eglm_ = np.log( eglm_ + log_c )
          fp_nat_ = np.log( fp_nat_ + log_c )

       ob = fp_nat_ - np.mean( eglm_, axis=0 )
       if DEBUG:
          ob = Him8_nat[::ng,::ng]  # debug    
          levs = np.arange( 20, 60, 5 ) 
          cmap = plt.cm.get_cmap("jet")

       x2d_ = x2d[ng2::ng,ng2::ng]
       y2d_ = y2d[ng2::ng,ng2::ng]
   
       i = 10
       j = 10   
   
       #nbin = 11
   
       mem_min = 32
       cnt = eglm_.shape[0]
   
       #hxmin = -10
       #hxmax = 10
   
       hxmin = -1
       hxmax = 12
   
       hymin = 0
       hymax = 300
   

       hxmin_ = -2.5
       hxmax_ = 12.5
       dh_ = 1.0
       if LOG:
          hxmin_ = -4.1
          hxmax_ = 4.1
          dh_ = 0.2

          hxmin = -4
          hxmax = 4

       nbin = int( ( hxmax_ - hxmin_ ) / dh_ )

       oerr = 1.0
  
   
       extend = 'both'
       norm = BoundaryNorm( levs, ncolors=cmap.N, clip=True )
 
       for j in range( y2d_.shape[0] ):
           for i in range( y2d_.shape[1] ):
               if fp_nat_[j,i] == 0.0 and fp_mem_[j,i] < mem_min:
                  continue

               if LOG and fp_nat_[j,i] < np.log( 0.01 + log_c ):
                  continue   

               std = np.std( eglm_[:,j,i], ddof=1 )
               h = 3.5 * std / np.power( cnt, 0.333 )
   
               fig, (ax1,ax2) = plt.subplots( 1, 2, figsize=( 10.5, 5))
               fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.9,
                                   wspace=0.2, hspace=0.3)
      
               #nbin = int( ( hxmax - hxmin ) / h )
   
               #var = fp_nat_[j,i] - eglm_[:,j,i] 
               var = eglm_[:,j,i] 
               print( "{0:0=3}, {1:0=3}, STD:{2:.4f}, nbin:{3:}".format(j,i, np.std( eglm_[:,j,i]), nbin ) )
               _, _, _ = ax1.hist( var, range=( hxmin_, hxmax_ ), bins=nbin,
                             rwidth=0.8, color='gray', alpha=0.5 )
           
               ax1.set_xlim( hxmin, hxmax )
               ax1.set_ylim( hymin, hymax )
    
               if LOG:
                  ax1.set_xticks( np.arange( -4, 5, 1 ) )
               else:
                  ax1.set_xticks( np.arange( 0, hxmax+2, 2) )
   
               ax1.vlines( x=fp_nat_[j,i], ymin=hymin, ymax=hymax,
                      ls="dashed", lw=2.5, 
                      label="Truth ({0:.2f})".format( fp_nat_[j,i] ), 
                      color='b', alpha=0.7 )
   
   #            ax1.vlines( x=[ fp_nat_[j,i]+oerr, fp_nat_[j,i]-oerr ], 
   #                   ymin=hymin, ymax=hymax,
   #                   ls="dashed", lw=0.5, 
   #                   color='b', alpha=0.7 )
   
               ax1.annotate(s='', xy=( fp_nat_[j,i]-oerr,150), xytext=(fp_nat_[j,i]+oerr,150), 
                   arrowprops=dict(arrowstyle='<->', color="b"))
   
               gues = np.mean(eglm_[:,j,i] )
               ax1.vlines( x=gues, ymin=hymin, ymax=hymax,
                      ls="dashed", lw=2.5, 
                      label="B ({0:.2f})".format( gues ), 
                      color='gray', alpha=0.8 )
                      
   #            ax1.vlines( x=[ gues-std, gues+std ], ymin=hymin, ymax=hymax,
   #                   ls="dashed", lw=0.5, 
   #                   color='gray', alpha=0.5 )
   
               ax1.annotate(s='', xy=( gues-std,100), xytext=(gues+std,100), 
                   arrowprops=dict(arrowstyle='<->', color="gray"))
   
               ax1.set_xlabel( "Flash count", fontsize=12 )
               ax1.set_ylabel( "Member", fontsize=12 )
   
               ax1.legend( fontsize=12, loc='upper right')
               #tit = '(x,y) = ({0:.0f}km, {1:.0f}km)\nfmem:{2:.0f}'.format( x2d_[j,i], y2d_[j,i], fp_mem_[j,i] )
               #fig.suptitle( tit, fontsize=13 )
 
               ptit = 'Background'
               ax1.text( 0.5, 1.01, ptit,
                        fontsize=13, transform=ax1.transAxes,
                        ha='center',
                        va='bottom', )


#               ob = read_evar4d_nc( INFO, vname="QG", tlev=1, typ="fcst", stime=INFO["time0"] )[0,10,::ng,::ng] * 1.e3
               SHADE = ax2.pcolormesh( x2d_, y2d_, ob, 
                         cmap=cmap, vmin=np.min(levs), vmax=np.max(levs),
                         norm=norm )
  
               off = ng*2.0/2
               ax2.plot( x2d_[j,i]+off, y2d_[j,i]+off, marker='*',
                       ms=10.0, color='gold' )

               pos = ax2.get_position()
               cb_h = pos.height * 0.9
               cb_w = 0.01 # pos.width * 0.8
               ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
               cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical', 
                                  ticks=levs, extend='both' )
               cb.ax.tick_params( labelsize=7 )
           
               xmin = 120
               xmax = 270
               ymin = 120
               ymax = 270
               ax2.set_xlim( xmin, xmax )
               ax2.set_ylim( ymin, ymax )
           
               ylab = "Y (km)"
               ax2.set_ylabel( ylab, fontsize=12)
           
               xlab = "X (km)"
               ax2.set_xlabel( xlab, fontsize=12)
           
               ax2.set_xticks( np.arange(xmin, xmax+20, 20) )
               ax2.set_yticks( np.arange(ymin, ymax+20, 20) )

               ptit = 'Obs-guess (flash member: {0:.0f})'.format( fp_mem_[j,i] )
               ax2.text( 0.5, 1.01, ptit,
                        fontsize=13, transform=ax2.transAxes,
                        ha='center',
                        va='bottom', )


               odir = "png/2p_obsguess_" + INFO["EXP"]

               if LOG:
                 ofig = 'LOG_2p_i{0:0=3}_j{1:0=3}_t{2:0=3}_ac{3:0=2}'.format( i, j, tlev, fp_acum )
               else:
                 ofig = '2p_i{0:0=3}_j{1:0=3}_t{2:0=3}_ac{3:0=2}'.format( i, j, tlev, fp_acum )
           
               print( ofig, odir )
    
               if not quick:
                  os.makedirs(odir, exist_ok=True)
                  plt.savefig(os.path.join(odir,ofig),
                              bbox_inches="tight", pad_inches = 0.1)
                  plt.clf()
                  plt.close('all')
               else:
                  plt.show()
           
   
   






    sys.exit()

#    fig, ax1 = plt.subplots(1, 1, figsize=( 5.5, 5))
#    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.95,
#                        wspace=0.4, hspace=0.2)
#
#
##    cmap = cmap_jet = plt.cm.get_cmap( "hot_r" )
##    levs = np.arange( 0, 5.0, 1.0 ) 
#
#
#    var = fp_nat-fp
#    SHADE = ax1.pcolormesh( x2d[::ng,::ng], y2d[::ng,::ng], var[::ng,::ng], 
#              cmap=cmap, vmin=np.min(levs), vmax=np.max(levs), )
#    print( np.max(fp) )
#
#    pos = ax1.get_position()
#    cb_h = pos.height * 0.9
#    cb_w = 0.01 # pos.width * 0.8
#    ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
#    cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'vertical', 
#                       ticks=levs, extend='max' )
#    cb.ax.tick_params( labelsize=6 )
#
#    xmin = 100
#    xmax = 300
#    ymin = 100
#    ymax = 300
#    ax1.set_xlim( xmin, xmax )
#    ax1.set_ylim( ymin, ymax )
#
#    ylab = "Y (km)"
#    ax1.set_ylabel( ylab, fontsize=12)
#
#    xlab = "X (km)"
#    ax1.set_xlabel( xlab, fontsize=12)
#
#    tit = "O-B {0:}".format( ctime.strftime('%H:%M:%S') )
#
#    fig.suptitle( tit, fontsize=14 )
#
#    plt.show()
#    sys.exit()


def write_nature_fp2d( INFO, tlev=0, fp_acum=1 ):

    fp = read_evar4d_nc( INFO, vname="FP", tlev=tlev, typ="fcst", stime=INFO["time0"] )[0,:,:,:]

    tmin = tlev - fp_acum + 1
    tmax = tlev

    for tlev_ in range( tmin, tmax ):
        fp += read_evar4d_nc( INFO, vname="FP", tlev=tlev_, typ="fcst", stime=INFO["time0"] )[0,:,:,:]
        print( tlev_)

    glm = get_GLM( fp, kernel )

    fn = os.path.join( INFO["GTOP"], INFO["EXP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), "fcst", 
                "fp_acum{0:0=2}.nc".format( fp_acum) )
    print( fn )
    nc = Dataset(fn, "w", format="NETCDF4")

    nc.createDimension("latitude", glm.shape[0])
    nc.createDimension("longitude", glm.shape[1])
    nc.createDimension("time", 1)

    XX = nc.createVariable("longitude","i4",("longitude",))
    XX.units = "degrees_east"

    YY = nc.createVariable("latitude","i4",("latitude",))
    YY.units = "degrees_north"


    times = nc.createVariable("time","f4",("time",))
    nc.description = "FP forcast from mean ({0:}-min accumlated)".format( int( fp_acum*INFO["DT"]/60 ) ) 

    times.units = "seconds since " + str( INFO["time0"] )
    times.calendar = "gregorian"
    times[0] = int( fp_acum*INFO["DT"] )

    XVAR = nc.createVariable("fp","f4",("latitude","longitude"))
    XVAR.units = ""


    XVAR[:,:] = glm[:]

    YY[:] = np.arange(1,glm.shape[0]+1)
    XX[:] = np.arange(1,glm.shape[1]+1)

    nc.close()

    sys.exit() 







def write_emean_fp2d( INFO, tlev=0, fp_acum=1 ):

    efp = read_evar4d_nc( INFO, vname="FP", tlev=tlev, typ="fcst", stime=INFO["time0"] )
    for tlev_ in range( 1, fp_acum ):
        efp += read_evar4d_nc( INFO, vname="FP", tlev=tlev_, typ="fcst", stime=INFO["time0"] )
        print( tlev_)

    eglm = get_eGLM( efp, kernel )

    eglm_mem = np.where( eglm > 0.0, 1.0, 0.0 )

    fmem = np.sum( eglm_mem[1:,:,:], axis=0 )
    mean = np.mean( eglm[1:,:,:], axis=0 )
    ens = eglm[1:,:,:]
    print( "chk", np.max(mean))

    fn = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), "fcst", 
                "mean_fp_acum{0:0=2}.nc".format( fp_acum) )
    print( fn )
    nc = Dataset(fn, "w", format="NETCDF4")

    nc.createDimension("latitude", mean.shape[0])
    nc.createDimension("longitude", mean.shape[1])
    nc.createDimension("time", 1)
    nc.createDimension("ensemble", ens.shape[0] )

    XX = nc.createVariable("longitude","i4",("longitude",))
    XX.units = "degrees_east"

    YY = nc.createVariable("latitude","i4",("latitude",))
    YY.units = "degrees_north"

    EE = nc.createVariable("ensemble","i4",("ensemble",))
    EE.units = ""

    times = nc.createVariable("time","f4",("time",))
    nc.description = "Ensemble mean FP ({0:}-min accumlated)".format( int( fp_acum*INFO["DT"]/60 ) ) 

    times.units = "seconds since " + str( INFO["time0"] )
    times.calendar = "gregorian"
    times[0] = int( fp_acum*INFO["DT"] )

    XVAR = nc.createVariable("fp","f4",("latitude","longitude"))
    XVAR.units = ""

    XVAR2 = nc.createVariable("fp_mem","f4",("latitude","longitude"))
    XVAR2.units = ""

    XVAR3 = nc.createVariable("efp","f4",("ensemble","latitude","longitude"))
    XVAR3.units = ""

    XVAR[:,:] = mean[:]
    XVAR2[:,:] = fmem[:]
    XVAR3[:,:] = ens[:]

    YY[:] = np.arange(1,mean.shape[0]+1)
    XX[:] = np.arange(1,mean.shape[1]+1)
    EE[:] = np.arange(1,ens.shape[0]+1)

    nc.close()







    sys.exit() 




       

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



EXP = "2000m_DA_0723"

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


tlev = 6  # 1800


fp_acum = 6
#fp_acum = 1

if fp_acum > tlev:
   fp_acum = tlev


#main( INFO, tlev=tlev, fp_acum=fp_acum )

# nature run
#INFO["EXP"] = "2000m_NODA_0723"
#INFO["time0"] = datetime( 2001, 1, 1, 1, 0, 0 )
#tlev = 12
#write_emean_fp2d( INFO, tlev=tlev, fp_acum=fp_acum )
#write_nature_fp2d( INFO, tlev=tlev, fp_acum=fp_acum )
main( INFO, tlev=tlev, fp_acum=fp_acum )
 
