import numpy as np
import sys
from datetime import datetime


from tools_LT import get_info, get_grads_all, get_evar4d


def main( exp, stime, typ , tlev, vname_list=["U"], nens=80 ):

   INFO = get_info( exp )
   if typ != "fcst":
      nens = 1

   for vname in vname_list:
      print(vname)
      var4d = get_evar4d(INFO, vname=vname, tlev=tlev, typ=typ,
                         stime=stime, member=nens )
      print("Done", vname, "t:", tlev, var4d.shape, np.max(var4d), np.min(var4d) )


exp = "2000m_DA_1022_FIR2km_N"
stime = datetime(2001,1,1,1,10) 
typ = "anal"
tmax = 2
nens = 80

typ = "fcst"
exp = "2000m_NODA_1022_FIR2km_N"
stime = datetime(2001,1,1,1,0) 
tmax = 13
nens = 1

typ = "fcst"

exp = "2000m_DA_0306"

#exp = "2000m_DA_0306_R_FP_DEBUG32_LOC90km_SINGLE"
#
#exp = "2000m_DA_0306_R_FP_DEBUG32_LOC90km_SINGLE0.1"
#exp = "2000m_DA_0306_R_FP_DEBUG32_LOC90km_SINGLE0.1_I98_J106"

exp = "2000m_DA_0306_FP_M32_LOC90km"


exp = "2000m_DA_0306_FP_M32_LOC90km_HT8"

exp = "2000m_DA_0306_FP_M32_LOC90km_QC5"

exp = "2000m_DA_0306_FP_M32_LOC90km_ZMAX23"

exp = "2000m_DA_0306_FP_M32_LOC90km"

#exp = "2000m_DA_0306_NOFP"
exp = "2000m_NODA_0306"


exp = "2000m_DA_0601"


exp = "2000m_DA_0601"

exp = "2000m_DA_0601_FP_M32_LOC90km"
exp = "2000m_DA_0601_FP_M01_LOC90km"
exp = "2000m_DA_0601_FP_M32_LOC90km_TEST"
exp = "2000m_DA_0601_FP_M32_LOC30km_TEST"
exp = "2000m_DA_0601_FP_M32_LOC30km_TEST2"
exp = "2000m_DA_0601_FP_M32_LOC30km"
exp = "2000m_DA_0601_FP_M32_LOC30km_LOG"
exp = "2000m_DA_0601_FP_M160_LOC30km"
exp = "2000m_DA_0601_FP_M160_LOC30km_POB"
exp = "2000m_DA_0601_FP_M160_LOC30km_NOB"
exp = "2000m_DA_0601_FP_M600_M32_LOC30km_NOB"
exp = "2000m_DA_0601_FP_M600_M160_LOC30km_NOB"
exp = "2000m_DA_0601_FP_M600_M160_LOC30km_NOB_LOG"
exp = "2000m_DA_0601_FP_GT"
exp = "2000m_DA_0601_FP_GT_POB"
#exp = "2000m_DA_0601_FP_GT_NOB"
#exp = "2000m_DA_0601_NOFP"

exp = "2000m_DA_0601"

exp = "2000m_DA_0723_FP"
#exp = "2000m_DA_0723_NOFP"
exp = "2000m_DA_0723_FP_M160"
exp = "2000m_DA_0723_FP_NOB"

exp = "2000m_DA_0723_FP_NOB_OBERR0.1"
exp = "2000m_DA_0723_FP_NOB_30km"

exp = "2000m_DA_0723"

exp = "2000m_DA_0723_FP_30min_NOB"
exp = "2000m_DA_0723_FP_30min"

exp = "2000m_DA_0723_NOFP_30min"

exp = "2000m_DA_0723_FP_30min_HT8"
exp = "2000m_DA_0723_FP_30min_M64"
exp = "2000m_DA_0723_FP_30min_NOB_M64"
exp = "2000m_DA_0723_FP_30min_M160"
exp = "2000m_DA_0723_FP_30min_M160_POB"

exp = "2000m_DA_0723_FP_30min_M160_GE3"

exp = "2000m_DA_0723_FP_30min_LOC30km"
exp = "2000m_NODA_0723"


exp = "2000m_DA_0723_FP_30min_LOC10km"
exp = "2000m_DA_0723_FP_30min_LOC30km_X175km_Y183km"

exp = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km_LOG"
exp = "2000m_DA_0723_FP_30min_LOC90km_X175km_Y183km"

exp = "2000m_DA_0723_FP_30min_LOC90km_LOC2D"

exp = "2000m_DA_0723_FP_30min_LOC10km_X167km_Y223km"

exp = "2000m_DA_0723_FP_30min_LOC10km"

exp = "2000m_DA_0723_FP_30min_LOC10km_LOG"

exp = "2000m_DA_0723_FP_30min_LOC10km_VLOCW20"

exp = "2000m_DA_0723_FP_30min_LOC10km_VLOC30km"

exp = "2000m_DA_0723_FP_30min_LOC30km_X183km_Y199km"

exp = "2000m_DA_0723_Z20km_FP_30min_LOC30km"
exp = "2000m_DA_0723_FP_30min_LOC30km_COERR0.2"
exp = "2000m_DA_0723_FP_30min_LOC20km"

stime = datetime(2001,1,1,1,40) 

 
stime = datetime(2001,1,1,1,30) 

stime = datetime( 2001, 1, 1, 1, 30 ) 

stime = datetime( 2001, 1, 1, 2, 0, 0 ) 
#stime = datetime( 2001, 1, 1, 1, 0 ) 

tmax = 19
tmax = 7
#tmax = 13
#tmax = 3
#nens = 80

#tmax = 1

nens = 0 # mean only
#nens = 320

typ = "fcst"

vname_list = ["U", "V", "W", "T", "P", "QV",
              "QC", "QR", "QI", "QS", "QG",
              "CC", "CR", "CI", "CS", "CG",
              "FP", "EX", "EY", "EZ",
              ]

#vname_list = ["FP", ]



for tlev in range( 0, tmax ):
   main( exp, stime, typ, tlev, vname_list=vname_list, nens=nens )
