import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt

def read_var( vname="QG", fn="" ):
    nc = Dataset(fn, "r", format="NETCDF4")

    var = nc.variables[vname][:]

    nc.close()
    return( var)

def main( nvar="G" ):

    if nvar == "G" or nvar == "S":
       v1 = "Q" + nvar
       v2 = "C" + nvar
    else:
       v1 = nvar
       v2 = nvar

    fn = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/2000m_DA_0723/20010101013000/fcst/{0:}_ens_t01800.nc".format( v1 )
    cfn = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/2000m_DA_0723/20010101013000/fcst/{0:}_ens_t01800.nc".format( v2 )
    
    eqg = read_var( vname=v1, fn=fn)
    ecg = read_var( vname=v2, fn=cfn)

    print( eqg.shape )

    cy = 230 // 2
    cy = 204 // 2
    var3d = eqg[1:,:,cy,:] * 1.e3
    gstd = np.std( var3d, axis=0, ddof=1)
    gmean = np.mean( var3d, axis=0, )

    ZDIM = 45
    DZ = 500.0
    Z = np.arange(DZ*0.5, DZ*ZDIM, DZ) * 1.e-3

    DX = 2000.0
    XDIM = 176
    X = np.arange( DX*0.5, DX*XDIM, DX ) * 1.e-3

    z2d, x2d = np.meshgrid( Z, X )

    fig, ((ax1, ax2, )) = plt.subplots(1, 2, figsize=( 12, 5))
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.92, top=0.92,
                        wspace=0.4, hspace=0.2)

    cmap = plt.cm.get_cmap("hot_r")
    levs = np.arange( 0, 5, 0.5 )

    c1 = ax1.contour( x2d, z2d, np.transpose( gstd ),
                      colors="k" )
    s1 = ax1.contourf( x2d, z2d, np.transpose( gmean ),
                       cmap=cmap, levels=levs  )

    plt.clabel( c1 )
    pos = ax1.get_position()
    cb_h = pos.height #0.01 #pos.height
    cb_w = 0.01
    ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )

    plt.colorbar( s1, cax=ax_cb )

    ax1.set_xlabel( "X (km)" )
    ax1.set_ylabel( "Height (km)" )

    if nvar == "G":
       tit1 = "Mean & Spread: graupel mixing ratio"
       tit2 = "Mean & Spread: graupel charge"
    elif nvar == "S":
       tit1 = "Mean & Spread: snow mixing ratio"
       tit2 = "Mean & Spread: snow charge"
    else:
       tit1 = "Spread: " + v1
       tit2 = tit1
    
    ax1.text( 0.5, 1.01, tit1, 
             fontsize=12, transform=ax1.transAxes,
             ha='center',
             va='bottom', )

    var3d = ecg[1:,:,cy,:]
    cstd = np.std( var3d, axis=0, ddof=1)
    cmean = np.mean( var3d, axis=0, ) 

    cmap_rb = plt.cm.get_cmap("RdBu_r")
    levs_rb = np.arange( -1, 1.2, 0.2 )
    c2 = ax2.contour( x2d, z2d, np.transpose( cstd ), 
                      colors='k' )
    plt.clabel( c2 )

    s2 = ax2.contourf( x2d, z2d, np.transpose( cmean ),
                       cmap=cmap_rb, levels=levs_rb )

    pos = ax2.get_position()
    cb_h = pos.height #0.01 #pos.height
    cb_w = 0.01
    ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
    plt.colorbar( s2, cax=ax_cb )

    ax2.set_xlabel( "X (km)" )
    ax2.set_ylabel( "Height (km)" )

    ax2.text( 0.5, 1.01, tit2, 
             fontsize=12, transform=ax2.transAxes,
             ha='center',
             va='bottom', )

    ax2.set_ylim( 1.0, 15.0 )
    ax1.set_ylim( 1.0, 15.0 )

    ax2.set_xlim( 100, 300 )
    ax1.set_xlim( 100, 300 )

    plt.show()
    

nvar = "G"
#nvar = "EZ"
#nvar = "EX"
#nvar = "EY"
nvar = "S"
main( nvar=nvar )
