import numpy as np
import os 
import sys

from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta

quick = True
#quick = False

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

def main( INFO, fp_acum=6, mem_min=32, single=False, sx=100, sy=100 ):


    top = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S') )

    ng = 4
    #ng2 = int( ng / 2 )
    ng2 = ng - 1


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

    fp_nat_ = fp_nat[ng2::ng,ng2::ng]

    

    
    dtop = os.path.join( top, "gues" )
    elt2d_g = read_lt2d_grads_e( dtop=dtop, emax=320, )

    dtop = os.path.join( top, "anal" )
    elt2d_a = read_lt2d_grads_e( dtop=dtop, emax=320, )

    print( elt2d_g.shape, fp_nat_.shape )
    print( np.max( elt2d_g ) )

    lxdim = elt2d_g.shape[2]
    lydim = elt2d_g.shape[1]

    import matplotlib.pyplot as plt

    for cx in range( lxdim ):
       for cy in range( lydim ):

          x_ = cx*8 + 7
          y_ = cy*8 + 7

          if single:
             if x_ != int( sx ) or y_ != int( sy ):
                continue

          gdat = elt2d_g[:,cy,cx]    
          adat = elt2d_a[:,cy,cx]    
          print( gdat.shape)     
          print( np.mean(gdat), np.mean(adat), fp_nat_[cy,cx], fp_nat_[cx,cy] )

          plt.contourf( np.mean(elt2d_g ,axis=0) )
          plt.show()

          plt.contourf( fp_nat_ )
          plt.show()

    sys.exit()






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
EXP = "2000m_DA_0723_FP_30min_LOC20km_X159km_Y231km"

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

single = True
sx = 159
sy = 231
main( INFO, fp_acum=fp_acum, mem_min=mem_min, single=single, sx=sx, sy=sy )
