import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta

import os
import sys

from tools_LT import write_radar_nc, write_Him8_nc, read_radar_grads, read_Him8_grads


def main( INFO, HIM8=True ):

    if HIM8:
       fn_Him8 = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], INFO["MEM"], 
                               "Him8_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".dat") 
       print( fn_Him8 )
       tbb = read_Him8_grads( fn_Him8, INFO )
       write_Him8_nc( INFO, tbb )


    fn_radar = os.path.join( INFO["TOP"], INFO["time0"].strftime('%Y%m%d%H%M%S'), INFO["TYPE"], INFO["MEM"], 
                       "radar_" + INFO["time0"].strftime('%Y%m%d%H%M%S_') + INFO["MEM"] + ".dat") 

    ref, vr = read_radar_grads( fn_radar, INFO )
    write_radar_nc( INFO, ref, vr )


###################

DX = 2000.0
DY = 2000.0
#XDIM = 192
#YDIM = 192
#ZDIM = 40
XDIM = 176
YDIM = 176
ZDIM = 45
TDIM = 13
TDIM = 7
#TDIM = 19
#TDIM = 3

#TDIM = 1 # anal

DZ = 500.0
DT = 300

X = np.arange( DX*0.5, DX*XDIM, DX )
Y = np.arange( DY*0.5, DY*YDIM, DY )
T = np.arange( 0, DT*TDIM, DT )
BAND = np.arange( 7, 17, 1 )

Z = np.arange(DZ*0.5, DZ*ZDIM, DZ)



EXP = "2000m_DA_0306"


EXP = "2000m_DA_0306_FP_M32_LOC90km"



EXP= "2000m_DA_0306_FP_M32_LOC30km"
EXP= "2000m_DA_0306_FP_M32_LOC150km"


EXP= "2000m_DA_0306_FP_M32_LOC90km"
#EXP = "2000m_DA_0306_NOFP"

EXP = "2000m_NODA_0306"
EXP = "2000m_NODA_0601"
EXP = "2000m_DA_0601"

EXP = "2000m_NODA_0723"
EXP = "2000m_DA_0723"

EXP= "2000m_DA_0723_NOFP"
EXP= "2000m_DA_0723_FP"
EXP = "2000m_DA_0723_FP_M160"

EXP = "2000m_DA_0723_FP_NOB"

EXP= "2000m_DA_0723_FP_NOB_OBERR0.1"
EXP = "2000m_DA_0723_FP_NOB_30km"

EXP = "2000m_DA_0723_FP_30min"
EXP = "2000m_DA_0723_FP_30min_NOB"

EXP = "2000m_DA_0723_NOFP_30min"

EXP = "2000m_DA_0723_FP_30min_HT8"
EXP = "2000m_DA_0723_FP_30min_M64"
EXP = "2000m_DA_0723_FP_30min_NOB_M64"

EXP = "2000m_DA_0723_FP_30min_M160"

EXP = "2000m_DA_0723_FP_30min_M160_POB"

EXP = "2000m_DA_0723_FP_30min_M160_GE3"
EXP = "2000m_DA_0723_FP_30min_LOC30km"

EXP = "2000m_DA_0723_FP_30min_LOC10km_HT16"

EXP = "2000m_DA_0723_FP_30min_LOC10km"

EXP = "2000m_DA_0723_FP_30min_LOC30km_X175km_Y183km"

EXP = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km_LOG"
EXP = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km"

EXP = "2000m_DA_0723_FP_30min_LOC90km_LOC2D"

EXP = "2000m_DA_0723_FP_30min_LOC10km_X167km_Y223km"

EXP = "2000m_DA_0723_FP_30min_LOC10km"
EXP = "2000m_DA_0723_FP_30min_LOC10km_LOG"

EXP = "2000m_DA_0723_FP_30min_LOC10km_VLOCW20"

EXP = "2000m_DA_0723_FP_30min_LOC10km_VLOC30km"

EXP = "2000m_DA_0723_FP_30min_LOC30km_X183km_Y199km"

EXP = "2000m_DA_0723_Z20km_FP_30min_LOC30km"

EXP = "2000m_DA_0723_FP_30min_LOC30km_COERR0.2"
EXP = "2000m_DA_0723_FP_30min_LOC20km"

TOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT/" + EXP
TYPE = "fcst"
#TYPE = "anal"

MEM = "mean"
#time0 = datetime( 2001, 1, 1, 1, 40, 0 )
time0 = datetime( 2001, 1, 1, 1, 20, 0 )
time0 = datetime( 2001, 1, 1, 1, 40, 0 )

time0 = datetime( 2001, 1, 1, 1, 0, 0 )
time0 = datetime( 2001, 1, 1, 1, 30, 0 )
time0 = datetime( 2001, 1, 1, 2, 0, 0 )

#time0 = datetime( 2001, 1, 1, 1, 5, 0 )
#time0_max = datetime( 2001, 1, 1, 2, 0, 0 )
time0_max = time0

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "TOP":TOP,
        "ZDIM":ZDIM, "Z":Z,
        "TYPE":TYPE, "MEM":MEM, "time0": time0 }

#print( INFO )
MEMBER = 320
mem_list = [str(x).zfill(4) for x in range(1,MEMBER+1)] 
mem_list = []
mem_list.append("mean")
print(mem_list)

HIM8 = True
HIM8 = False

while time0 <= time0_max:

   INFO["time0"] = time0
   for mem in mem_list:
      INFO["MEM"] = mem
      main( INFO, HIM8=HIM8 )

   time0 += timedelta( minutes=5 )

