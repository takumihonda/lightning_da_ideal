import numpy as np
from netCDF4 import Dataset

import matplotlib.colors as mpc

import sys
import os
from datetime import datetime
from datetime import timedelta

Rdry = 287.04

def get_rmse_bias3d_wv( var=np.zeros(1), rvar=np.zeros(1) ):

    wthrs = 5.0
    var1 = np.where( var > wthrs, 1.0, 0.0)
    rvar1 = np.where( rvar > wthrs, 1.0, 0.0)

    dif = var1 - rvar1 

    err = np.mean( np.sum( var1 ) - np.sum( rvar1 ) )
    err1d = np.sum( dif, axis=(1,2) ) 

    bias = err
    bias1d = err1d

    return( err, err1d, bias, bias1d )

def get_rmse_bias3d( var=np.zeros(1), rvar=np.zeros(1) ):

    dif2 = np.square( var - rvar )  

    err = np.sqrt( np.mean( dif2 ) )
    err1d = np.sqrt( np.mean( dif2, axis=(1,2) ) )

    bias = np.mean( var - rvar )
    bias1d = np.mean( var - rvar, axis=(1,2) )

    return( err, err1d, bias, bias1d )

def get_flash_tot(nc, slice_info):

    try:
      FLASH_TOT = read_ncvar(nc, "FLASH", slice_info)
    except:
      FLASH_TOT = read_ncvar(nc, "PosFLASH", slice_info)
      FLASH_TOT += read_ncvar(nc, "NegFLASH", slice_info)

    return(FLASH_TOT)

def get_qcrg_tot(nc, slice_info):

    try:
      QCRG_TOT = read_ncvar(nc, "QCRG_TOT", slice_info)
    except:
      QCRG_TOT = read_ncvar(nc, "QCRG_C", slice_info)
      QCRG_TOT += read_ncvar(nc, "QCRG_R", slice_info)
      QCRG_TOT += read_ncvar(nc, "QCRG_I", slice_info)
      QCRG_TOT += read_ncvar(nc, "QCRG_S", slice_info)
      QCRG_TOT += read_ncvar(nc, "QCRG_G", slice_info)

    return(QCRG_TOT)

def get_qh(nc, slice_info):

    try:
      QHYD = read_ncvar(nc, "QHYD", slice_info)
    except:
      QHYD = read_ncvar(nc, "QC", slice_info)
      QHYD += read_ncvar(nc, "QR", slice_info)
      QHYD += read_ncvar(nc, "QI", slice_info)
      QHYD += read_ncvar(nc, "QS", slice_info)
      QHYD += read_ncvar(nc, "QG", slice_info)

    return(QHYD)

def get_vort(nc, slice_info, DIMS):
    U = read_ncvar(nc, "U", slice_info)
    V = read_ncvar(nc, "V", slice_info)

    dudy, dudx = np.gradient(U[0,0,:,:], 2)
    dvdy, dvdx = np.gradient(V[0,0,:,:], 2)

    dudx = dudx * 0.5 / DIMS["dx"]
    dvdx = dvdx * 0.5 / DIMS["dx"]

    dudy = dudy * 0.5 / DIMS["dy"]
    dvdy = dvdy * 0.5 / DIMS["dy"]

    return(dvdx - dudy)

def get_dims(nc):

    x = nc.variables["x"][:]
    y = nc.variables["y"][:]
    z = nc.variables["z"][:]
    time = nc.variables["time"][:]

    DIMS = {"x":x, "y":y, "z":z, "time":time, "dx":x[2]-x[1], "dy":y[2]-y[1]}

    return(DIMS)

def read_Encvar_mean(member, path, nvar, slice_info):

    tlevs = slice_info[0]
    zlevs = slice_info[1]
    ylevs = slice_info[2]
    xlevs = slice_info[3]

    emean = np.zeros((tlevs[1] - tlevs[0],
                      zlevs[1] - zlevs[0],
                      ylevs[1] - ylevs[0],
                      xlevs[1] - xlevs[0]))

    for m in range(1,member+1):
       mem = str(m).zfill(4)
       fn = path + "/history_" + mem + ".nc"
       nc = Dataset(fn, 'r', format='NETCDF4')

       if nvar == "DBZ":
         DBZ = get_ncvar(nc, nvar, slice_info) 
         emean[:,:,:,:] += np.where(DBZ > 0.0, DBZ, 0.0) / np.float64(member)
       else:
         emean[:,:,:,:] += get_ncvar(nc, nvar, slice_info) / np.float64(member)
       nc.close()

    return(emean)

def read_Encvar(member, path, nvar, slice_info):

    tlevs = slice_info[0]
    zlevs = slice_info[1]
    ylevs = slice_info[2]
    xlevs = slice_info[3]

    evar = np.zeros((member,
                     tlevs[1] - tlevs[0],
                     zlevs[1] - zlevs[0],
                     ylevs[1] - ylevs[0],
                     xlevs[1] - xlevs[0]))

    for m in range(1,member+1):
       mem = str(m).zfill(4)
       fn = path + "/history_" + mem + ".nc"
       nc = Dataset(fn, 'r', format='NETCDF4')

       evar[m-1,:,:,:,:] = get_ncvar(nc, nvar, slice_info)
       nc.close()

    return(evar)

def read_ncvar(nc, nvar, slice_info):
    
    tlevs = slice_info[0]
    zlevs = slice_info[1]
    ylevs = slice_info[2]
    xlevs = slice_info[3]

    var = nc.variables[nvar][tlevs[0]:tlevs[1],\
                             zlevs[0]:zlevs[1],\
                             ylevs[0]:ylevs[1],\
                             xlevs[0]:xlevs[1]]

# FLASH variables are NOT accumlated (after 05/22/2019)
# 
#    if nvar == "FLASH" or nvar == "PosFLASH" or nvar == "NegFLASH":
#      try:
#        var -= nc.variables[nvar][tlevs[0]-1:tlevs[1]-1,\
#                                  zlevs[0]:zlevs[1],\
#                                  ylevs[0]:ylevs[1],\
#                                  xlevs[0]:xlevs[1]]
#      except:
#        print("!Make sure FLASH!")
#
    return(var)

def get_ncvar(nc, nvar, slice_info):

    if nvar == "QHYD":
       var = get_qh(nc, slice_info)

    elif nvar == "QCRG_TOT":
       var = get_qcrg_tot(nc, slice_info)

    elif nvar == "DBZ":
       var = get_dbz(nc, slice_info)

    elif nvar == "VORT":
       var = get_vort(nc, slice_info)

    elif nvar == "FLASH":
       var = get_flash_tot(nc, slice_info)

    elif nvar == "FLASH2D" or nvar == "PosFLASH2D" or nvar == "NegFLASH2D":
       var = get_flash_tot(nc, slice_info)
       slice_info_col = np.copy(slice_info)
       slice_info_col[1] = [0,-1]
       var = np.sum(get_flash_tot(nc, slice_info_col), axis=1, keepdims=True)
    else:
       var = read_ncvar(nc, nvar, slice_info)

    return(var)

def get_dbz(nc, slice_info):


#    if "DBZ" in nc.variables.keys():
#       DBZ = read_ncvar(nc, "DBZ", slice_info)
#
#       if not np.max(DBZ) >= 0.0 and np.max(DBZ) < 100.0 and not np.isnan(np.max(DBZ)):
#          return(DBZ)

    MAXF = 0.5
   
    QR = read_ncvar(nc, "QR", slice_info)
    QS = read_ncvar(nc, "QS", slice_info)
    QG = read_ncvar(nc, "QG", slice_info)
   
    Fg = np.where((QR > 0.0) & (QG > 0.0), \
                  MAXF * np.power(np.minimum(QR/QG, QG/QR), 1.0/3.0), \
                  0.0)
   
    Fwg = np.where((QR > 0.0) & (QG > 0.0), \
                   QR / (QR + QG), \
                   0.0)
   
    Fs = np.where((QR > 0.0) & (QS > 0.0), \
                  MAXF * np.power(np.minimum(QR/QS, QS/QR), 1.0/3.0), \
                  0.0)
   
    Fws = np.where((QR > 0.0) & (QS > 0.0), \
                   QR / (QR + QS), \
                   0.0)
   
    QMS = Fs * (QR + QS) # melting snow
    QMG = Fg * (QR + QG) # melting hail
   
    QR = (1.0 - Fs - Fg) * QR
    QS = (1.0 - Fs) * QS
    QG = (1.0 - Fg) * QG
   
    # rain
    a_r = 2.53*1.e4
    b_r = 1.84
   
    # snow
    a_s = 3.48*1.e3
    b_s = 1.66
   
    # graupel # Tomita (2008) for X-band (Amemiya et al., in prep)
    a_g = 5.54*1.e3
    b_g = 1.7
   
    DENS = get_rho(nc, slice_info)
  
    DBZ = np.where(QR > 0.0, a_r * np.power(DENS * QR * 1.e3, b_r), 0.0)
    DBZ += np.where(QS > 0.0, a_s * np.power(DENS * QS * 1.e3, b_s), 0.0)
    DBZ += np.where(QG > 0.0, a_g * np.power(DENS * QG * 1.e3, b_g), 0.0)
   
    ZMS = np.where(QMS > 0.0, (0.00491 + 5.75 * Fws - 5.588 * np.square(Fws)) * 1.e5 * \
                  np.power(DENS * QMS * 1.e3, 1.67 - 0.202 * Fws + 0.398 * np.square(Fws)), 0.0)
   
    ZMG = np.where(QMG > 0.0, (0.809 + 10.13 * Fwg - 5.98 * np.square(Fwg)) * 1.e5 * \
                  np.power(DENS * QMG * 1.e3, 1.48 + 0.0448 * Fwg - 0.0313 * np.square(Fwg)), 0.0)
   
    DBZ += ZMS + ZMG
   
    DBZ = np.where(DBZ > 0.0, 10*np.log10(DBZ), 0.0)
 
  
    return(DBZ)
   
def get_rho(nc, slice_info):
 
    RHO = read_ncvar(nc, "QV", slice_info)
    RHO = (1.0 + 0.61 * RHO) * read_ncvar(nc, "T", slice_info) # virtual temperature
    RHO = read_ncvar(nc, "PRES", slice_info) / Rdry / RHO

    return(RHO)

def def_cmap(nvar):

    if nvar == "QHYD" or nvar == "VORT" or nvar == "QG" or nvar == "QS" or nvar == "QI" or nvar == "QH" or nvar == "QV":
      slevs = np.array([0.1, 0.5, 1, 2, 3, 4, 5.0, 6, 7, 8,10,12])
      cmap = mpc.ListedColormap(['cyan','dodgerblue', 'b',
                                 'lime', 'limegreen','forestgreen','yellow','gold',
                                 'orange','red', 'magenta',
                                 'purple'])

    elif nvar == "DBZ":
      slevs = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
      cmap = mpc.ListedColormap(['cyan','dodgerblue', 
                                 'lime', 'limegreen','yellow',
                                 'orange', 'red', 'firebrick', 'magenta',
                                 'purple'])

    elif nvar == "QCRG_TOT" or nvar == "QCRG_S" or nvar == "QCRG_G" or nvar == "QCRG_I" or \
         nvar == "QCRG" or nvar == "OEP":

      slevs = np.array([  -1.5, -1.2, -0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5])
      cmap = mpc.ListedColormap(['darkblue', 'royalblue','cornflowerblue','lightskyblue','lightcyan',
                                 'w',
                                 'mistyrose','salmon','crimson','firebrick','maroon',
                                 ])


    elif nvar == "CG" or nvar == "CS" or nvar == "CR" or nvar == "CC" or nvar == "CI":

      slevs = np.array([  -4, -3, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 3, 4])
      cmap = mpc.ListedColormap(['darkblue', 'royalblue','cornflowerblue','lightskyblue','lightcyan',
                                 'w',
                                 'mistyrose','salmon','crimson','firebrick','maroon',
                                 ])

    elif nvar == "INC":

      slevs = np.array([ -8, -6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6, 8])
      cmap = mpc.ListedColormap(['darkblue', 'royalblue','cornflowerblue','lightskyblue','lightcyan',
                                 'w',
                                 'mistyrose','salmon','crimson','firebrick','maroon',
                                 ])

    elif nvar == "W" or nvar == "V" or nvar == "U":
      cmap = mpc.ListedColormap(['darkblue', 'royalblue','cornflowerblue','lightskyblue','lightcyan',
                                 'w',
                                 'mistyrose','salmon','crimson','firebrick','maroon',
                                 ])

      slevs = np.array([-32, -24, -16, -8, -4, -1, 1, 4, 8, 16, 24, 32])

    elif nvar == "VR":
      cmap = mpc.ListedColormap(['purple','dodgerblue','cyan','limegreen','lime',
                                 'w',
                                 'yellow','orange', 'red', 'firebrick','magenta',
                                 ])

      slevs = np.array([-40, -32, -24, -16, -8, -4, 4, 8, 16, 24, 32, 40])

    elif nvar == "FLASH" or nvar == "PosFLASH" or nvar == "NegFLASH" or \
         nvar == "FlashPoint" or nvar == "FLASH2D" or nvar == "LT" or nvar == "FP":

      slevs = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 6])
      cmap = mpc.ListedColormap(['cyan','dodgerblue', 
                                 'lime', 'limegreen','yellow',
                                 'orange', 'red', 'firebrick', 'magenta',
                                 'purple'])

      if nvar == "FLASH2D":
        slevs = np.array([1, 2, 4, 8, 12, 16, 20, 24])

    elif nvar == "LTP":
      slevs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ])
      cmap = mpc.ListedColormap(['cyan','dodgerblue', 
                                 'lime', 'limegreen','yellow',
                                 'orange', 'red', 'firebrick', 'magenta',
                                 'purple'])


    elif nvar == "COR":
      slevs = np.array([-0.8, -0.6, -0.4, -0.3,-0.2, 0.2, 0.3, 0.4, 0.6, 0.8])
      cmap = mpc.ListedColormap(['darkblue','cornflowerblue','lightskyblue','lightcyan',
                                 'w',
                                 'mistyrose','salmon','firebrick','maroon',
                                 ])

    elif nvar == "T":
      slevs = np.array([230, 238, 246, 254, 262, 270, 278, 286, 294])
      cmap = mpc.ListedColormap(['darkblue','cornflowerblue','lightskyblue','lightcyan',
                                 'mistyrose','salmon','firebrick','maroon',
                                 ])

    elif nvar == "EX" or nvar == "EY" or nvar == "EZ" or "EABS":
      slevs = np.array([-200,-160,-120, -80,-40, 40, 80, 120, 160, 200])

      cmap = mpc.ListedColormap(['darkblue','cornflowerblue','lightskyblue','lightcyan',
                                 'w',
                                 'mistyrose','salmon','firebrick','maroon',
                                 ])

    cnorm = mpc.BoundaryNorm(slevs, ncolors=len(slevs)-1, clip=False)

    if nvar == "LTP" or nvar == "DBZ" or nvar == "QHYD" or nvar == "QG" or \
       nvar == "QS" or nvar == "QI" or nvar == "QH" or nvar == "QV" or \
       nvar == "VORT" or nvar == "FP":
      cmap.set_under('w', alpha=1.0)
    else:
      cmap.set_under('gray', alpha=1.0)
      cmap.set_over('gray', alpha=1.0)

    return(cmap, cnorm, slevs)

def ecor(member, path, nvar, nvar1d, slice_info, slice_info1p):
    
    # Get Ensmeble var
    E_VAR = read_Encvar(member, path, nvar, slice_info)

    E_VAR1d = read_Encvar(member, path, nvar1d, slice_info1p)

    E_COR = np.zeros(E_VAR.shape[1:])
 
    for m in range(1, member+1):
       E_COR[:,:,:,:] += (E_VAR[m-1,:,:,:,:] - np.mean(E_VAR,axis=0)) \
                       * (E_VAR1d[m-1,:,:,:,:] - np.mean(E_VAR1d,axis=0)) \
                         
    E_COR = E_COR / np.float64(member - 1) \
                  / np.std(E_VAR,axis=0,ddof=1) \
                  / np.std(E_VAR1d,axis=0,ddof=1) 

    return(E_COR)

def get_info(exp, FCST=True):

    T0 = 3600.0 #sec (FT at stime)
    DT = 300 # sec
 
    DX = 2000.0
    DY = 2000.0
    DZ = 500.0

#    XDIM = 192
#    YDIM = 192
    XDIM = 176
    YDIM = 176
    X = np.arange(DX*0.5, DX*XDIM, DX)
    Y = np.arange(DY*0.5, DY*YDIM, DY)

#    ZDIM = 40
    ZDIM = 45
    Z = np.arange(DZ*0.5, DZ*ZDIM, DZ)
    GTOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT"
    print( exp )


    TOP = os.path.join(GTOP, exp)
    INFO = {"XDIM":XDIM, "YDIM":YDIM, "ZDIM":ZDIM, "X":X, "Y":Y, "Z":Z,
            "tdim": 121, "stime": datetime(2001, 1, 1, 1, 0), "DT":DT, "T0":T0,
            "TOP":TOP, "GTOP":GTOP, "EXP":exp,
            "U":1, "V":2, "W":3, "T":4, "P":5, 
            "QV":6, "QC":7, "QR":8, "QI":9, "QS":10, "QG":11, 
            "PFL":12, "NFL":13, "FP":14, "LTP":15, 
            "CC":16, "CR":17, "CI":18, "CS":19, "CG":20, 
            "EX":21, "EY":22, "EZ":23,
            "NVAR":23, 
            "RADARNVAR":2, "DBZ":1, "VR":2,
            "HIM8NVAR":10, "TBB13":7,
           }
    if not FCST:
       INFO["NVAR"] = 22
       INFO["CC"] = 12
       INFO["CR"] = 13
       INFO["CI"] = 14
       INFO["CS"] = 15
       INFO["CG"] = 16
       INFO["OEP"] = 17
       INFO["NC"] = 18
       INFO["NR"] = 19
       INFO["NI"] = 20
       INFO["NS"] = 21
       INFO["NG"] = 22



    return(INFO) 
    
def get_grads_conv_lt(INFO, vname="U", tlev=1, m=0, typ="fcst", stime=datetime(2001,1,1,1,0,0)):

    #print("Get grads")
    mem = str(m).zfill(4)
    if m == 0:
       mem = "mean"
    elif m == -1:
       mem = "sprd"

    fhead = "all3d_"
    nvar = INFO["NVAR"]
    if vname == "DBZ" or vname == "VR":
       fhead = "radar_"
       nvar = INFO["RADARNVAR"]
    elif vname == "TBB13":
       fhead = "Him8_"
       nvar = INFO["HIM8NVAR"]

    if typ == "fcst":
       if stime is None:
          fname =  fhead + INFO["stime"].strftime('%Y%m%d%H%M%S_') + mem + ".dat"
          ctime = INFO["stime"].strftime('%Y%m%d%H%M%S')
       else:
          fname =  fhead + stime.strftime('%Y%m%d%H%M%S_') + mem + ".dat"
          ctime = stime.strftime('%Y%m%d%H%M%S')

    elif typ == "anal" or typ == "gues":
       fname =  "init_grads.dat"
       ctime = (INFO["stime"] + timedelta(seconds=int(INFO["DT"]*tlev))).strftime('%Y%m%d%H%M%S')
       if vname == "DBZ" or vname == "VR":
         fname =  "radar_" + ctime + "_" + mem + ".dat"
       tlev = 0

    fn = os.path.join(INFO["TOP"], ctime, typ, mem,
         fname)

    try:
       infile = open(fn)
    except:
       print("Failed to open")
       print(fn)
       sys.exit()
 
    cnt2d = INFO["XDIM"] * INFO["YDIM"]
    cnt3d = cnt2d * INFO["ZDIM"]

    rec = (nvar * tlev + (INFO[vname] - 1)) * cnt3d
#    if vname == "LT":
#       #     flash3d + flash2d + flashpoint3d + flashpoint2d
#       rec = tlev * (2*cnt3d + 2*cnt2d) 
#    elif vname == "FP":
#       #     flash3d + flash2d + flashpoint3d + flashpoint2d
#       rec = tlev * (2*cnt3d + 2*cnt2d) + cnt3d 

    infile.seek(rec*4)
    tmp3d = np.fromfile(infile, dtype=np.dtype('<f4'), count=cnt3d)  #little endian
    var3d = np.reshape(tmp3d, (INFO["ZDIM"], INFO["YDIM"], INFO["XDIM"]))

    if typ is not "fcst":
       if vname == "CC" or vname == "CR" or vname == "CI" or vname == "CS" or vname == "CG":
          var3d = var3d * 1.e-6 # fC => nC

    #print(vname,np.max(var3d),fn)
    infile.close()

    return(var3d)


def get_grads_all(INFO, vname="U", tlev=1, m=0, typ="fcst", stime=datetime(2001,1,1,1,0,0)):

    if vname == "QH":
       var3d = get_grads_conv_lt(INFO, vname="QC", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="QR", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="QI", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="QS", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="QG", tlev=tlev, m=m, typ=typ, stime=stime)

    elif vname == "QCRG":
       var3d = get_grads_conv_lt(INFO, vname="CC", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="CR", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="CI", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="CS", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="CG", tlev=tlev, m=m, typ=typ, stime=stime)

    elif vname == "LT":
       var3d = get_grads_conv_lt(INFO, vname="PFL", tlev=tlev, m=m, typ=typ, stime=stime)
       var3d += get_grads_conv_lt(INFO, vname="NFL", tlev=tlev, m=m, typ=typ, stime=stime)

    elif vname == "EABS":
       var3d = np.square(get_grads_conv_lt(INFO, vname="EX", tlev=tlev, m=m, typ=typ, stime=stime))
       var3d += np.square(get_grads_conv_lt(INFO, vname="EY", tlev=tlev, m=m, typ=typ, stime=stime))
       var3d += np.square(get_grads_conv_lt(INFO, vname="EZ", tlev=tlev, m=m, typ=typ, stime=stime))
       var3d = np.sqrt(var3d)

    elif vname == "RHO":
       
       tv = get_grads_conv_lt(INFO, vname="T", tlev=tlev, m=m, typ=typ, stime=stime) * \
            ( 1.0 + 0.61 * get_grads_conv_lt(INFO, vname="QV", tlev=tlev, m=m, typ=typ, stime=stime) )
       var3d = get_grads_conv_lt(INFO, vname="P", tlev=tlev, m=m, typ=typ, stime=stime) / ( tv * Rdry )

    else:
       var3d = get_grads_conv_lt(INFO, vname=vname, tlev=tlev, m=m, typ=typ, stime=stime)

    return(var3d)


def get_evar4d(INFO, vname="U", tlev=1, typ="fcst", stime=datetime(2001,1,1,1,0,0), member=50):

       print( "tentative" )
#    try:
#       return(read_evar4d_nc(INFO, vname=vname, tlev=tlev, typ=typ, stime=stime))
#    except:   
       print("read from grads")

       var3d = get_grads_all(INFO, vname=vname, tlev=tlev, m=0, typ=typ, stime=stime)
   
       # member + 1 (mean)
       var4d = np.zeros((member+1,var3d.shape[0],var3d.shape[1],var3d.shape[2]))
   
       var4d[0,:,:,:] = var3d[:,:,:]
   
       for m in range(1, member+1):
           var4d[m,:,:,:] = get_grads_all(INFO, vname=vname, tlev=tlev, m=m, typ=typ, stime=stime)
  
       write_evar4d_nc(var4d, INFO, vname=vname, tlev=tlev, typ=typ, stime=stime)
       return(var4d)


def read_evar4d_nc(INFO, vname="U", tlev=1, typ="fcst", stime=datetime(2001,1,1,1,0,0)):
    print("read_evar4d")

    ctime = stime.strftime('%Y%m%d%H%M%S')

    fn = os.path.join(INFO["GTOP"], INFO["EXP"], ctime, typ, vname + "_ens_t" + str(INFO["DT"] * tlev).zfill(5) + ".nc")

    print(fn)

    nc = Dataset(fn, "r", format="NETCDF4")

    var4d = nc.variables[vname][:]

    nc.close()

    return(var4d)

def write_evar4d_nc(var4d, INFO, vname="U", tlev=1, typ="fcst", stime=datetime(2001,1,1,1,0,0)):
    print("write_evar4d")

    ctime = stime.strftime('%Y%m%d%H%M%S')

    fn = os.path.join(INFO["GTOP"], INFO["EXP"], ctime, typ, vname + "_ens_t" + str(INFO["DT"] * tlev).zfill(5) + ".nc")

    print(fn)

    nc = Dataset(fn, "w", format="NETCDF4")

    nc.createDimension("latitude", var4d.shape[2])
    nc.createDimension("longitude", var4d.shape[3])
    nc.createDimension("level", var4d.shape[1])
    nc.createDimension("time", 1)
    nc.createDimension("ensemble", var4d.shape[0])

    XX = nc.createVariable("longitude","i4",("longitude",))
    XX.units = "degrees_east"

    YY = nc.createVariable("latitude","i4",("latitude",))
    YY.units = "degrees_north"

    ZZ = nc.createVariable("level","i4",("level",))
    ZZ.units = ""

    EE = nc.createVariable("ensemble","i4",("ensemble",))
    EE.units = ""

    times = nc.createVariable("time","f4",("time",))
    nc.description = "Ensemble " + vname 

    times.units = "seconds since " + str(stime)
    times.calendar = "gregorian"
    times[0] = INFO["DT"]*tlev

    XVAR = nc.createVariable(vname,"f4",("ensemble","level","latitude","longitude"))
    XVAR.units = ""

    XVAR[:,:,:,:] = var4d[:]
    EE[:] = np.arange(1,var4d.shape[0]+1)
    ZZ[:] = np.arange(1,var4d.shape[1]+1)
    YY[:] = np.arange(1,var4d.shape[2]+1)
    XX[:] = np.arange(1,var4d.shape[3]+1)

    nc.close()

def write_radar_nc( INFO, z4d, vr4d ):

    fn = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"],
                       INFO["MEM"], "radar_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".nc") 
    print( fn )

    nc = Dataset( fn, "w", format="NETCDF4" ) 
    nc.createDimension( "latitude", z4d.shape[2] )
    nc.createDimension( "longitude", z4d.shape[3] )
    nc.createDimension( "level", z4d.shape[1] )
    nc.createDimension( "time", z4d.shape[0] )

    XX = nc.createVariable( "longitude","i4",("longitude",) )
    XX.units = "degrees_east"

    YY = nc.createVariable( "latitude","i4",("latitude",) )
    YY.units = "degrees_north"

    ZZ = nc.createVariable( "level","f4",("level",) )
    ZZ.units = ""

    times = nc.createVariable( "time","f4",("time",) )
    times.units = "seconds since " + str( INFO["time0"] )
    times.calendar = "gregorian"

    XVAR_Z = nc.createVariable( "z","f4",("time","level","latitude","longitude") )
    XVAR_Z.units = "dBZ"

    XVAR_VR = nc.createVariable( "vr","f4",("time","level","latitude","longitude") )
    XVAR_VR.units = "m/s"

    XVAR_Z[:,:,:,:] = z4d[:]
    XVAR_VR[:,:,:,:] = vr4d[:]
    ZZ[:] = INFO["Z"][:]
    YY[:] = INFO["Y"][:]
    XX[:] = INFO["X"][:]
    times[:] = INFO["T"][:]

    nc.close()

def write_Him8_nc( INFO, var4d ):

    fn = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"],
                       INFO["MEM"], "Him8_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".nc") 
    print( fn )

    nc = Dataset( fn, "w", format="NETCDF4" ) 
    nc.createDimension( "latitude", var4d.shape[2] )
    nc.createDimension( "longitude", var4d.shape[3] )
    nc.createDimension( "band", var4d.shape[1] )
    nc.createDimension( "time", var4d.shape[0] )

    XX = nc.createVariable( "longitude","i4",("longitude",) )
    XX.units = "degrees_east"

    YY = nc.createVariable( "latitude","i4",("latitude",) )
    YY.units = "degrees_north"

    ZZ = nc.createVariable( "band","i4",("band",) )
    ZZ.units = ""

    times = nc.createVariable( "time","f4",("time",) )
    times.units = "seconds since " + str( INFO["time0"] )
    times.calendar = "gregorian"

    XVAR = nc.createVariable( "tbb","f4",("time","band","latitude","longitude") )
    XVAR.units = ""

    XVAR[:,:,:,:] = var4d[:]
    ZZ[:] = INFO["BAND"][:]
    YY[:] = INFO["Y"][:]
    XX[:] = INFO["X"][:]
    times[:] = INFO["T"][:]

    nc.close()

def read_radar_grads( fn, INFO ):

    try:
       infile = open( fn )
    except:
       print( "Failed to open" )
       sys.exit()

    cnt2d = INFO["XDIM"] * INFO["YDIM"]
    cnt3d = cnt2d * INFO["ZDIM"] 

    var_z = np.zeros( ( INFO["TDIM"], INFO["ZDIM"], INFO["YDIM"], INFO["XDIM"] ) )
    var_vr = np.zeros( ( INFO["TDIM"], INFO["ZDIM"], INFO["YDIM"], INFO["XDIM"] ) )
    print( var_z.shape )

    for t in range(INFO["TDIM"]):
       rec = cnt3d * 2 * t # z + vr
       infile.seek(rec*4)
       tmp3d = np.fromfile( infile, dtype=np.dtype('<f4'), count=cnt3d )
       var3d = np.reshape( tmp3d, ( INFO["ZDIM"], INFO["YDIM"], INFO["XDIM"] ) )
       var_z[t,:,:,:] = var3d[:]

       rec = cnt3d * 2 * t + cnt3d # z + vr
       infile.seek(rec*4)
       tmp3d = np.fromfile( infile, dtype=np.dtype('<f4'), count=cnt3d )
       var3d = np.reshape( tmp3d, ( INFO["ZDIM"], INFO["YDIM"], INFO["XDIM"] ) )
       var_vr[t,:,:,:] = var3d[:]

    return( var_z, var_vr )

def read_Him8_grads( fn, INFO ):

    try:
       infile = open( fn )
    except:
       print( "Failed to open" )
       sys.exit()

    cnt2d = INFO["XDIM"] * INFO["YDIM"]
    cnt3d = cnt2d * INFO["NBAND"] 
    cnt4d = cnt3d * INFO["TDIM"]

    tmp4d = np.fromfile( infile, dtype=np.dtype('<f4'), count=cnt4d )
    var = np.reshape( tmp4d, ( INFO["TDIM"], INFO["NBAND"], INFO["YDIM"], INFO["XDIM"] ) )
    #print( var4d.shape )
    #print( var4d[:,40,40])

    infile.close()

    return( var )

def read_evar_only( INFO, tlev=0, vname="QG" ):
    # read ens variables
    try:
       evar = read_evar4d_nc( INFO, vname=vname, tlev=tlev, typ="fcst", stime=INFO["time0"] )
    except:
       if vname is "QCRG":
          evar = read_evar4d_nc( INFO, vname="CC", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CI", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          write_evar4d_nc(evar, INFO, vname="QCRG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
       elif vname is "QHYD":
          evar = read_evar4d_nc( INFO, vname="QC", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QI", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          write_evar4d_nc(evar, INFO, vname="QHYD", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    return( evar )

def read_evars( INFO, tlev=0, vname="QG", member=80 ):
    # Read variables

    # Him8
    try:
       etbb = read_evar4d_nc( INFO, vname="TBB", tlev=tlev, typ="fcst", stime=INFO["time0"] )
    except:
       fn_Him8 = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], "mean",
                               "Him8_" + INFO["time0"].strftime('%Y%m%d%H%M%S_mean') + ".nc")
       print( fn_Him8 )
       nc = Dataset(fn_Him8, 'r', format='NETCDF4')
       tbb = nc.variables["tbb"][tlev,:,:,:]
       nc.close()
       etbb = np.zeros( (member+1,tbb.shape[0],tbb.shape[1],tbb.shape[2]) )
       etbb[0,:,:,:] = tbb[:]

       for m in range(1, member+1):
           mem = str(m).zfill(4)
           fn_Him8 = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], mem,
                                  "Him8_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + mem + ".nc")

           nc = Dataset(fn_Him8, 'r', format='NETCDF4')
           tbb_ = nc.variables["tbb"][tlev,:,:,:]
           nc.close()
           etbb[m,:,:,:] = tbb_[:]

       write_evar4d_nc(etbb, INFO, vname="TBB", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    # Radar
    try:
       ez = read_evar4d_nc( INFO, vname="Z", tlev=tlev, typ="fcst", stime=INFO["time0"] )
#       evr = read_evar4d_nc( INFO, vname="VR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
    except:
       fn_radar = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], "mean",
                          "radar_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + "mean" + ".nc")
       nc = Dataset(fn_radar, 'r', format='NETCDF4')
       z = nc.variables["z"][tlev,:,:,:]
#       vr = nc.variables["vr"][tlev,:,:,:]
       nc.close()

       ez = np.zeros( (member+1,z.shape[0],z.shape[1],z.shape[2]) )
#       evr = np.zeros( (member+1,vr.shape[0],vr.shape[1],vr.shape[2]) )
       ez[0,:,:,:] = z[:]
#       evr[0,:,:,:] = vr[:]

       for m in range(1, member+1):
           mem = str(m).zfill(4)
           fn_radar = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], mem,
                              "radar_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + mem + ".nc")

           nc = Dataset(fn_radar, 'r', format='NETCDF4')
           z_ = nc.variables["z"][tlev,:,:,:]
#           vr_ = nc.variables["vr"][tlev,:,:,:]
           nc.close()

           ez[m,:,:,:] = z_[:]
#           evr[m,:,:,:] = vr_[:]

       write_evar4d_nc(ez, INFO, vname="Z", tlev=tlev, typ="fcst", stime=INFO["time0"] )
#       write_evar4d_nc(evr, INFO, vname="VR", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    # Flash point
    efp = read_evar4d_nc( INFO, vname="FP", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    # Electric field
#    ee = read_evar4d_nc( INFO, vname="EZ", tlev=tlev, typ="fcst", stime=INFO["time0"] )

#    try:
#       ee = read_evar4d_nc( INFO, vname="EABS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
#    except:
#       ee = np.square( read_evar4d_nc( INFO, vname="EX", tlev=tlev, typ="fcst", stime=INFO["time0"] ) )
#       ee += np.square( read_evar4d_nc( INFO, vname="EY", tlev=tlev, typ="fcst", stime=INFO["time0"] ) )
#       ee += np.square( read_evar4d_nc( INFO, vname="EZ", tlev=tlev, typ="fcst", stime=INFO["time0"] ) )
#
#       ee = np.sqrt( ee )
#       write_evar4d_nc(ee, INFO, vname="EABS", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    # read ens variables
    try:
       evar = read_evar4d_nc( INFO, vname=vname, tlev=tlev, typ="fcst", stime=INFO["time0"] )
    except:
       if vname == "QCRG":
          evar = read_evar4d_nc( INFO, vname="CC", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CI", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          write_evar4d_nc(evar, INFO, vname="QCRG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
       elif vname == "QHYD":
          evar = read_evar4d_nc( INFO, vname="QC", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QI", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          write_evar4d_nc(evar, INFO, vname="QHYD", tlev=tlev, typ="fcst", stime=INFO["time0"] )
       elif vname == "EABS":
         ee = np.square( read_evar4d_nc( INFO, vname="EX", tlev=tlev, typ="fcst", stime=INFO["time0"] ) )
         ee += np.square( read_evar4d_nc( INFO, vname="EY", tlev=tlev, typ="fcst", stime=INFO["time0"] ) )
         ee += np.square( read_evar4d_nc( INFO, vname="EZ", tlev=tlev, typ="fcst", stime=INFO["time0"] ) )

         ee = np.sqrt( ee )
         write_evar4d_nc(ee, INFO, vname="EABS", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    return( etbb, ez, efp, evar )

def get_ecor1d( var1d, evar ):

    print(evar.shape)
    nmem = evar.shape[0] - 1

    var1d -= np.mean( var1d[1:] )
    if np.max(var1d) == 0.0:
       ecor1d = evar[1,:] * np.nan
       return( ecor1d )

    evar -= np.mean( evar[1:,:], axis=0 )
    ecor1d = var1d[1] * evar[1,:]
    for m in range( 2, nmem+1 ):
       ecor1d += var1d[m] * evar[m,:]

    ecor1d = ecor1d / np.std( var1d[1:], ddof=1, axis=0 ) / np.std( evar[1:,:], ddof=1, axis=0 ) / ( nmem )

    return( ecor1d )

def get_ecor( var1d, evar ):

    nmem = evar.shape[0] - 1

    var1d -= np.mean( var1d[1:] )
    if np.max(var1d) == 0.0:
       ecor = np.zeros( evar[1,:,:,:].shape )
       #ecor = evar[1,:,:,:] * np.nan
       return( ecor )

    evar -= np.mean( evar[1:,:,:,:], axis=0 )
    ecor = var1d[1] * evar[1,:,:,:]
    for m in range( 2, nmem+1 ):
       ecor += var1d[m] * evar[m,:,:,:]

    ecor = ecor / np.std( var1d[1:], ddof=1, axis=0 ) / np.std( evar[1:,:,:,:], ddof=1, axis=0 ) / ( nmem )

    return( ecor )

def get_GLM( fp, kernel ):

    if fp.ndim == 3:
       fp_ = np.sum( fp, axis=0 )
    elif fp.ndim == 2:
       fp_ = fp

    from scipy import ndimage
    glm = ndimage.convolve( fp_, kernel, mode='reflect' )  # GLM

    return( glm )

def get_eGLM( efp, kernel ):

    if efp.ndim == 3:
       efp_ = efp
    elif efp.ndim == 4:
       efp_ = np.sum( efp, axis=1 )

    emem = efp_.shape[0]
    from scipy import ndimage
    glm = ndimage.convolve( efp_[0,:,:], kernel, mode='reflect' )  # GLM
   
    eglm = np.zeros( (emem, efp_.shape[1], efp_.shape[2]) )
    eglm[0,:,:] = glm[:,:]
    for m in range( 1, emem ):
       eglm[m,:,:] = ndimage.convolve( efp_[m,:,:], kernel, mode='reflect' )  # GLM
   
    return( eglm )

def read_evar_only( INFO, tlev=0, vname="QG", member=80 ):
    # read ens variables
    try:
       evar = read_evar4d_nc( INFO, vname=vname, tlev=tlev, typ="fcst", stime=INFO["time0"] )
    except:
       if vname is "QCRG":
          evar = read_evar4d_nc( INFO, vname="CC", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CI", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="CG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          write_evar4d_nc(evar, INFO, vname="QCRG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
       elif vname is "QHYD":
          evar = read_evar4d_nc( INFO, vname="QC", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QR", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QI", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QS", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          evar += read_evar4d_nc( INFO, vname="QG", tlev=tlev, typ="fcst", stime=INFO["time0"] )
          write_evar4d_nc(evar, INFO, vname="QHYD", tlev=tlev, typ="fcst", stime=INFO["time0"] )

    return( evar )

def setup_12p():
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure( figsize=(12.5, 8.5) ) # h:v
    gs = gridspec.GridSpec(105, 155) # v:h

    fig.subplots_adjust( left = 0.05, right=0.99, top=0.94, bottom=0.08 )

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

    hmin2 = hmin1_r + dh_r + 2*pdh
    hmin2_r = hmin2 + dh + pdh
    ax2 = plt.subplot(   gs[vmax1:vmax1+dv,hmin2:hmin2+dh] )
    ax2_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin2:hmin2+dh] )
    ax2_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin2_r:hmin2_r+dh_r] )

    hmin3 = hmin2_r + dh_r + 2*pdh
    hmin3_r = hmin3 + dh + pdh
    ax3 = plt.subplot(   gs[vmax1:vmax1+dv,hmin3:hmin3+dh] )
    ax3_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin3:hmin3+dh] )
    ax3_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin3_r:hmin3_r+dh_r] )

    vmax4_t = vmax1 + dv + 2*pdv + pdv
    vmax4 = vmax4_t + dv_t + pdv
    ax4 = plt.subplot(   gs[vmax4:vmax4+dv,hmin1:hmin1+dh] )
    ax4_t = plt.subplot( gs[vmax4_t:vmax4_t+dv_t,    hmin1:hmin1+dh] )
    ax4_r = plt.subplot( gs[vmax4:vmax4+dv,  hmin1_r:hmin1_r+dh_r] )

    ax5 = plt.subplot(   gs[vmax4:vmax4+dv,       hmin2:hmin2+dh] )
    ax5_t = plt.subplot( gs[vmax4_t:vmax4_t+dv_t, hmin2:hmin2+dh] )
    ax5_r = plt.subplot( gs[vmax4:vmax4+dv,       hmin2_r:hmin2_r+dh_r] )

    ax6 = plt.subplot(   gs[vmax4:vmax4+dv,       hmin3:hmin3+dh] )
    ax6_t = plt.subplot( gs[vmax4_t:vmax4_t+dv_t, hmin3:hmin3+dh] )
    ax6_r = plt.subplot( gs[vmax4:vmax4+dv,       hmin3_r:hmin3_r+dh_r] )

    ax_l = [ ax1, ax2, ax3, ax4, ax5, ax6, ax1_r, ax1_t, ax2_r, ax2_t, ax3_r, ax3_t,
             ax4_r, ax4_t, ax5_r, ax5_t, ax6_r, ax6_t ]

    crs_l = [ "XY", "XY", "XY", "XY", "XY", "XY", 
              "XZ", "ZY", "XZ", "ZY", "XZ", "ZY",
              "XZ", "ZY", "XZ", "ZY", "XZ", "ZY", ]


    return( ax_l, crs_l, fig )

def setup_9p():
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure( figsize=(12.5, 8.5) ) # h:v
    gs = gridspec.GridSpec(105, 155) # v:h

    fig.subplots_adjust( left = 0.05, right=0.99, top=0.94, bottom=0.08 )

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

    hmin2 = hmin1_r + dh_r + 2*pdh
    hmin2_r = hmin2 + dh + pdh
    ax2 = plt.subplot(   gs[vmax1:vmax1+dv,hmin2:hmin2+dh] )
    ax2_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin2:hmin2+dh] )
    ax2_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin2_r:hmin2_r+dh_r] )

    hmin3 = hmin2_r + dh_r + 2*pdh
    hmin3_r = hmin3 + dh + pdh
    ax3 = plt.subplot(   gs[vmax1:vmax1+dv,hmin3:hmin3+dh] )
    ax3_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin3:hmin3+dh] )
    ax3_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin3_r:hmin3_r+dh_r] )

    vmax4_t = vmax1 + dv + 2*pdv + pdv + 10
    vmax4 = vmax4_t #+ pdv
#    ax4 = plt.subplot(   gs[vmax4:vmax4+dv+pdv+dv_t,hmin1:hmin1+60] )
#    ax5 = plt.subplot(   gs[vmax4:vmax4+dv+pdv+dv_t,       85:145] )

    ax4 = plt.subplot(   gs[vmax4:vmax4+40,  0: 35] )
    ax5 = plt.subplot(   gs[vmax4:vmax4+dv+pdv+dv_t, 55: 90] )
    ax6 = plt.subplot(   gs[vmax4:vmax4+dv+pdv+dv_t, 110:145] )

    ax_l = [ ax1, ax2, ax3, ax1_r, ax1_t, ax2_r, ax2_t, ax3_r, ax3_t,
             ax4, ax5, ax6,
             ]

    crs_l = [ "XY", "XY", "XY", 
              "XZ", "ZY", "XZ", "ZY", "XZ", "ZY",
              "TZ", "TZ", "TZ",
              ]


    return( ax_l, crs_l, fig )

def get_dat( INFO, rEXP="2000m_NODA_0306", EXP1="2000m_DA_0306", vname1="QHYD", time0=datetime(2001,1,1,1,0) ):



    tmin = 0
    tmax = 7

    ofn = "fcst_err_bias_{0:}_{1:}_{2:}_time{3:}.npz".format( vname1, EXP1, rEXP, time0.strftime('%H%M%S') )
    odir = "dat/fcst_{0:}".format( EXP1 )
    print( ofn, odir )

    vname1_ = vname1

    fac = 1.0
    if vname1 == "QR" or vname1 == "QS" or vname1 == "QG" or \
       vname1 == "QI" or vname1 == "QHYD" or vname1 == "QV":
       fac = 1.e3


    try:
       data = np.load( odir + "/" + ofn, allow_pickle=False )
       err_t = data["err"]
       err1d_t = data["err1d"]
       bias_t = data["bias"]
       bias1d_t = data["bias1d"]
    except:
       print( "failed" )

       err_t = np.zeros( (tmax - tmin) )
       err1d_t = np.zeros( (tmax - tmin, INFO["ZDIM"]) )
       bias_t = np.zeros( (tmax - tmin) )
       bias1d_t = np.zeros( (tmax - tmin, INFO["ZDIM"]) )

       for idx, tlev in enumerate( range(tmin, tmax) ):

           # reset
           INFO["time0"] = time0

           ctime = INFO["time0"] + timedelta(seconds=INFO["DT"]*tlev )

           INFO["EXP"] = EXP1
           INFO["MEM"] = "mean"
           INFO["TYPE"] = "fcst"

           print("CHECK", INFO["time0"], tlev )
           #tbb_exp1, z_exp1, vr_exp1 = read_vars( INFO, tlev=tlev, HIM8=False )
           evar_exp1 = read_evar_only( INFO, tlev=tlev, vname=vname1_ )
           efp_exp1 = read_evar_only( INFO, tlev=tlev, vname="FP" )


           ft_sec =  int( INFO["DT"]*tlev )


           # nature run
           # read variables
           INFO["EXP"] = rEXP #"2000m_NODA_0306"
           INFO["MEM"] = "mean"
           INFO["TYPE"] = "fcst"
           INFO["time0"] = datetime(2001, 1, 1, 1, 0)
           tlev_nat = int( ( ctime - datetime(2001, 1, 1, 1, 0) ).total_seconds() / INFO["DT"] )
           tbb_nat, z_nat, vr_nat = read_vars( INFO, tlev=tlev_nat, HIM8=False )
           print( "DEBUG", tlev_nat, ctime)
           efp_nat = read_evar_only( INFO, tlev=tlev_nat, vname="FP" )
           ew_nat = read_evar_only( INFO, tlev=tlev_nat, vname="W" )
           evar_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname1_ )


           err, err1d, bias, bias1d = get_rmse_bias3d( var=evar_exp1[0,:,:,:], rvar=evar_nat[0,:,:,:] )
           print("evars: ", evar_nat.shape, evar_exp1.shape, err, err1d.shape )


           err_t[idx] = err
           err1d_t[idx,:] = err1d[:]
           bias_t[idx] = bias
           bias1d_t[idx,:] = bias1d[:]

       os.makedirs( odir, exist_ok=True )
       np.savez( odir + "/" + ofn, err=err_t, err1d=err1d_t, bias=bias_t, bias1d=bias1d_t)



    return( err_t*fac, err1d_t*fac, bias_t*fac, bias1d_t*fac )

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







def get_dat_volume( INFO, rEXP="2000m_NODA_0306", EXP1="2000m_DA_0306", vname1="WV", time0=datetime(2001,1,1,1,0) ):



    tmin = 0
    tmax = 7

    ofn = "fcst_err_bias_{0:}_{1:}_{2:}_time{3:}.npz".format( vname1, EXP1, rEXP, time0.strftime('%H%M%S') )
    odir = "dat/fcst_{0:}".format( EXP1 )
    print( ofn, odir )

    vname1_ = vname1
    if vname1 == "WV":
       vname1_ = "W"
    elif vname1 == "GV":
       vname1_ = "QG"


    fac = 1.0


    try:
       data = np.load( odir + "/" + ofn, allow_pickle=False )
       err_t = data["err"]
       bias_t = data["bias"]
    except:
       print( "failed" )

       err_t = np.zeros( (tmax - tmin) )
#       err1d_t = np.zeros( (tmax - tmin, INFO["ZDIM"]) )
       bias_t = np.zeros( (tmax - tmin) )
#       bias1d_t = np.zeros( (tmax - tmin, INFO["ZDIM"]) )

       for idx, tlev in enumerate( range(tmin, tmax) ):

           # reset
           INFO["time0"] = time0

           ctime = INFO["time0"] + timedelta(seconds=INFO["DT"]*tlev )

           INFO["EXP"] = EXP1
           INFO["MEM"] = "mean"
           INFO["TYPE"] = "fcst"

           print("CHECK", INFO["time0"], tlev )
           #tbb_exp1, z_exp1, vr_exp1 = read_vars( INFO, tlev=tlev, HIM8=False )
           evar_exp1 = read_evar_only( INFO, tlev=tlev, vname=vname1_ )
           efp_exp1 = read_evar_only( INFO, tlev=tlev, vname="FP" )


           ft_sec =  int( INFO["DT"]*tlev )


           # nature run
           # read variables
           INFO["EXP"] = rEXP #"2000m_NODA_0306"
           INFO["MEM"] = "mean"
           INFO["TYPE"] = "fcst"
           INFO["time0"] = datetime(2001, 1, 1, 1, 0)
           tlev_nat = int( ( ctime - datetime(2001, 1, 1, 1, 0) ).total_seconds() / INFO["DT"] )
           tbb_nat, z_nat, vr_nat = read_vars( INFO, tlev=tlev_nat, HIM8=False )
           print( "DEBUG", tlev_nat, ctime)
           efp_nat = read_evar_only( INFO, tlev=tlev_nat, vname="FP" )
           ew_nat = read_evar_only( INFO, tlev=tlev_nat, vname="W" )
           evar_nat = read_evar_only( INFO, tlev=tlev_nat, vname=vname1_ )

           wthrs = 5.0
           if vname1_ == "QG":
              wthrs = 0.001

           rvar = np.where( evar_nat[0,:,:,:] > wthrs, 1.0, 0.0)
           var = np.where( evar_exp1[0,:,:,:] > wthrs, 1.0, 0.0)
           err = np.abs( np.sum( var ) - np.sum( rvar ) )
           bias = np.sum( var ) - np.sum( rvar )

           print( "Score\n", err, bias, "\n" )

#           err, err1d, bias, bias1d = get_rmse_bias3d( var=evar_exp1[0,:,:,:], rvar=evar_nat[0,:,:,:] )
#           print("evars: ", evar_nat.shape, evar_exp1.shape, err, err1d.shape )


           err_t[idx] = err
#           err1d_t[idx,:] = err1d[:]
           bias_t[idx] = bias
#           bias1d_t[idx,:] = bias1d[:]

       os.makedirs( odir, exist_ok=True )
       np.savez( odir + "/" + ofn, err=err_t, bias=bias_t )



    return( err_t*fac, bias_t*fac )

