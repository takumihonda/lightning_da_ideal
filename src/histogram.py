import numpy as np
import os 
import sys

from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta

quick = True
quick = False

def read_lt2d_grads( fn="" ):

    infile = open( fn )

    lxdim = 44
    lydim = 44

    cnt2d = lxdim * lydim
    tmp2d = np.fromfile( infile, dtype=np.dtype('<f4'), count=cnt2d )
    var2d = np.reshape( tmp2d, (lydim, lxdim) )

    return( var2d )


def read_lt2d_grads_e( dtop="", emax=320, ):

    for m in range( emax ):
       m4 = str( m+1 ).zfill(4)
       fn = os.path.join( dtop, m4, "init_lt2d.dat" )
       var2d_ = read_lt2d_grads( fn=fn )
       
       if m == 0:
          elt2d = np.zeros( ( emax, var2d_.shape[0], var2d_.shape[1] ) )
          elt2d[0,:,:] = var2d_[:]
       else:
          elt2d[m,:,:] = var2d_[:]

    return( elt2d )

def get_obs2d( INFO, time=datetime(2001,1,1,1,0), ng=4 ):
    fn = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/2000m_NODA_0723_30min/20010101010000/fcst/mean/obs/" + time.strftime('fp_%Y%m%d%H%M%S.dat')

    infile = open( fn )

    gx = INFO["XDIM"]
    gy = INFO["YDIM"]
    gz = INFO["ZDIM"]

    cnt2d = gx*gy
    cnt3d = gx*gy*gz
    infile.seek(cnt3d*2*4) # skip two 3D variables( fp3d & err3d)

    tmp2d = np.fromfile( infile, dtype=np.dtype('<f4'), count=cnt2d )
    fp2d = np.reshape( tmp2d, (gy, gx) )

    tmp2d = np.fromfile( infile, dtype=np.dtype('<f4'), count=cnt2d )
    err2d = np.reshape( tmp2d, (gy, gx) )

    OBSERR_FP_TRUE = 1.0
    err2d_ = np.where( fp2d < 1.0e-6, 0.0, err2d*OBSERR_FP_TRUE )

    ng2 = ng - 1

    return( fp2d[ng2::ng,ng2::ng] + err2d_[ng2::ng,ng2::ng] )

def main( INFO, fp_acum=6, mem_min=32, single=False, sx=100, sy=100 ):

    top = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S') )
    exp_ = INFO["EXP"]

    print( top )

    obs2d = get_obs2d( INFO, time=INFO["time0"], ng=4 )
#    print( "DEBUG" )
#    print( np.min( obs2d ) )
#    print( obs2d[ obs2d < 0.0 ] )
#    print( len( obs2d[ obs2d > 0.0 ] ) )
#    print( obs2d.size )
#    sys.exit()

#    print( obs2d.shape )
#    sys.exit()
#
#    # nature run
#    INFO["EXP"] = "2000m_NODA_0723"
#    INFO["time0"] = datetime( 2001, 1, 1, 1, 0, 0 )
#    tlev = 12
#
#    ctime = INFO["time0"] + timedelta( seconds=INFO["DT"]*tlev )
#
#    fn_nat = os.path.join( INFO["GTOP"], INFO["EXP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), "fcst",
#                  "fp_acum{0:0=2}.nc".format( fp_acum) )
#    print( fn_nat )
#
#    try:
#       nc = Dataset( fn_nat, 'r', format='NETCDF4')
#       fp_nat = nc.variables["fp"][:,:]
#       nc.close()
#    except:
#       print( "Failed to get natunre run")
#       print( fn_nat )
#       sys.exit()
#
#    ng = 4
#    ng2 = int( ng / 2 )
#    ng2 = ng - 1
#
#    fp_nat_ = fp_nat[ng2::ng,ng2::ng]
#
 
    fp_nat_ = obs2d
   
    dtop = os.path.join( top, "gues" )
    elt2d_g = read_lt2d_grads_e( dtop=dtop, emax=320, )

    dtop = os.path.join( top, "anal" )
    elt2d_a = read_lt2d_grads_e( dtop=dtop, emax=320, )

    print( elt2d_g.shape )
    print( np.max( elt2d_g ) )

    lxdim = elt2d_g.shape[2]
    lydim = elt2d_g.shape[1]

    import matplotlib.pyplot as plt

    xmin = -4
    xmax = 10
    nbin = ( xmax - xmin ) + 1

    ymin = 0
    ymax = 300

    cnt_g = np.where( elt2d_g >= 1.0, 1.0, 0.0 )
    cnt2d_g = np.sum( cnt_g, axis=0 )

#    print( np.max( cnt2d_g ), cnt2d_g.shape  )
#
#    cmap = plt.cm.get_cmap("RdBu_r")
#    cmap.set_over('gray', alpha=1.0)
#    cmap.set_under('gray', alpha=1.0)
#    levs = np.arange( -4, 4.5, 0.5 )
#    from matplotlib.colors import BoundaryNorm
#    norm = BoundaryNorm( levs, ncolors=cmap.N, clip=True )
#    plt.pcolormesh( fp_nat_-np.mean( elt2d_g, axis=0 ),
#                  cmap=cmap, vmin=np.min(levs), vmax=np.max(levs),
#                  norm=norm )
#    plt.show()
#    sys.exit()

    for cx in range( lxdim ):
       for cy in range( lydim ):

          x_ = cx*8 + 7
          y_ = cy*8 + 7

          if single:
             if x_ != int( sx ) or y_ != int( sy ):
                continue

          gdat = elt2d_g[:,cy,cx]    
          adat = elt2d_a[:,cy,cx]    
          

          std_g = np.std( gdat, axis=0, ddof=1 )
          std_a = np.std( adat, axis=0, ddof=1 )

          if cnt2d_g[cy,cx] < mem_min:
             continue
          print( "std: {0:.2f}, {1:.2f}, cnt:{2:}, obs{3:.2f}".format( std_g, std_a, int( cnt2d_g[cy,cx] ), fp_nat_[cy,cx] ) )

          fig, (ax1,ax2) = plt.subplots( 1, 2, figsize=( 10.5, 5 ) )
          fig.subplots_adjust( left=0.06, bottom=0.1, right=0.94, top=0.9,
                          wspace=0.2, hspace=0.3 )
          ax_l = [ ax1, ax2 ]
          dat_l = [ gdat, adat ]

          tit_l = [ "guess", "anal" ]

          tit = "LT2d (x={0:.0f}km, y={1:.0f}km), fmem:{2:.0f}".format( x_, y_, cnt2d_g[cy,cx] )
          fig.suptitle( tit, fontsize=12 )

          c_l = [ "gray", "r" ]
          alp_l = [ 0.5, 0.5 ]

          xlab = "Flash"
          ylab = "Member"

          err_reduction = np.abs(  fp_nat_[cy,cx] - np.mean( dat_l[0] ) ) \
                         -np.abs(  fp_nat_[cy,cx] - np.mean( dat_l[1] ) ) 

          err_reduction_rate = err_reduction / np.abs(  fp_nat_[cy,cx] - np.mean( dat_l[0] ) )

          for i, ax in enumerate( ax_l ):
              mean = np.mean( dat_l[i] )
              std = np.std( dat_l[i], ddof=1 )

              if i == 1:
                 ax.hist( dat_l[0], range=( xmin, xmax), bins=nbin,
                        align='mid', color=c_l[0], alpha=alp_l[i],
                        )

                 ax.vlines( ymin=ymin, ymax=ymax, x=np.mean( dat_l[0] ), ls='dashed', 
                         color=c_l[0] )

                 note = r'$\sigma_b$: {0:.2f}'.format( np.std( dat_l[0], ddof=1), )
                 ax.annotate( s='', xy=( np.mean( dat_l[0] )-np.std( dat_l[0], ddof=1) ,80), 
                              xytext=( np.mean( dat_l[0] )+np.std( dat_l[0], ddof=1), 80 ), 
                              arrowprops=dict(arrowstyle='<->', color=c_l[0], 
                              label=note ) )

              ax.hist( dat_l[i], range=( xmin, xmax), bins=nbin,
                     align='mid', color=c_l[i], alpha=alp_l[i],
                     )
           
              ax.set_ylim( ymin, ymax )

              note = 'Mean: {0:.2f}'.format( mean, )
              ax.vlines( ymin=ymin, ymax=ymax, x=mean, ls='dashed', 
                         color=c_l[i], label=note )

              ax.annotate( s='', xy=(mean-std, 50), 
                           xytext=( mean+std, 50 ), 
                           arrowprops=dict(arrowstyle='<->', color=c_l[i] ) )

              note = r'$\sigma$: {0:.2f}'.format( std, )
              ax.text( 0.99, 0.8, note,
                       fontsize=11, transform=ax.transAxes,
                       ha='right',
                       va='top', )

              if i == 1:
                 note = 'Error reduction:{0:+.2f}\nRate:{1:+.0f}%'.format( err_reduction, err_reduction_rate*100 )
                 ax.text( 0.99, 0.7, note,
                          fontsize=11, transform=ax.transAxes,
                          ha='right',
                          va='top', )


              ptit = 'Background'
              ax.text( 0.5, 1.01, tit_l[i],
                       fontsize=12, transform=ax.transAxes,
                       ha='center',
                       va='bottom', )

              ax.set_xlabel( xlab, fontsize=10 )
              ax.set_ylabel( ylab, fontsize=10 )

              # obs
              note = 'Obs: {0:.2f}'.format( fp_nat_[cy,cx], )
              ax.vlines( ymin=ymin, ymax=ymax, x=fp_nat_[cy,cx], ls='solid', 
                         color='b', label=note  )

              ax.legend( loc='upper right', fontsize=12 )

          odir = "png/hist_{0:}".format( exp_ )
          ofig = "2p_hist_lt2d_x{0:0=3}_y{1:0=3}.png".format( cx, cy)
          print( ofig )
          if not quick:
             os.makedirs( odir, exist_ok=True )
             plt.savefig( os.path.join( odir, ofig ),
                         bbox_inches="tight", pad_inches = 0.1)
             plt.clf()
             plt.close('all')
          else:
             plt.show()


################

DX = 2000.0
DY = 2000.0
TDIM = 13

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



EXP = "2000m_DA_0723_FP_30min_LOC20km"
EXP = "2000m_DA_0723_FP_30min_LOC20km_M240"
#EXP = "2000m_DA_0723_FP_30min_LOC20km_X159km_Y207km"
#EXP = "2000m_DA_0723_FP_30min_LOC20km_X159km_Y231km"

TOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/" + EXP
GTOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT"
TYPE = "fcst"
time00 = datetime( 2001, 1, 1, 0, 0, 0 )

time0 = datetime( 2001, 1, 1, 2, 0, 0 )

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "TOP":TOP, "GTOP":GTOP,
        "ZDIM":ZDIM, "Z":Z, "DT":DT,
        "TYPE":TYPE, "EXP":EXP,
        "time0": time0, "time00": time00  }


fp_acum = 6
mem_min = 32
mem_min = 240

single = True
single = False
sx = 159
sy = 207
main( INFO, fp_acum=fp_acum, mem_min=mem_min, single=single, sx=sx, sy=sy )
