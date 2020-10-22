import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt

def read_var( vname="QG", fn="" ):
    nc = Dataset(fn, "r", format="NETCDF4")

    var = nc.variables[vname][:]

    nc.close()
    return( var)

def main(  ):
    fn = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/2000m_DA_0723/20010101013000/fcst/{0:}_ens_t01800.nc".format( "QG" )
    cfn = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/2000m_DA_0723/20010101013000/fcst/{0:}_ens_t01800.nc".format( "CG" )
    
    eqg = read_var( vname="QG", fn=fn)
    ecg = read_var( vname="CG", fn=cfn)

    print( eqg.shape )

    cy = 230 // 2
    var3d = eqg[1:,:,cy,:] * 1.e3
    gstd = np.std( var3d, axis=0, ddof=1)

    ZDIM = 45
    DZ = 500.0
    Z = np.arange(DZ*0.5, DZ*ZDIM, DZ) * 1.e-3

    DX = 2000.0
    XDIM = 176
    X = np.arange( DX*0.5, DX*XDIM, DX ) * 1.e-3

    z2d, x2d = np.meshgrid( Z, X )

    fig, ((ax1, ax2, )) = plt.subplots(1, 2, figsize=( 9, 5))
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.92, top=0.92,
                        wspace=0.4, hspace=0.2)


    s1 = ax1.contourf( x2d, z2d, np.transpose( gstd ) )

    pos = ax1.get_position()
    cb_h = pos.height #0.01 #pos.height
    cb_w = 0.01
    ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )

    plt.colorbar( s1, cax=ax_cb )

    ax1.set_xlabel( "X (km)" )
    ax1.set_ylabel( "Height (km)" )

    ax1.text( 0.5, 1.01, "Spread: graupel mixing ratio",
             fontsize=12, transform=ax1.transAxes,
             ha='center',
             va='bottom', )

    var3d = ecg[1:,:,cy,:]
    cstd = np.std( var3d, axis=0, ddof=1)
    s2 = ax2.contourf( x2d, z2d, np.transpose( cstd ) )

    pos = ax2.get_position()
    cb_h = pos.height #0.01 #pos.height
    cb_w = 0.01
    ax_cb = fig.add_axes( [pos.x1+0.01, pos.y0, cb_w, cb_h] )
    plt.colorbar( s2, cax=ax_cb )

    ax2.set_xlabel( "X (km)" )
    ax2.set_ylabel( "Height (km)" )

    ax2.text( 0.5, 1.01, "Spread: graupel charge",
             fontsize=12, transform=ax2.transAxes,
             ha='center',
             va='bottom', )

    plt.show()
    

main()
