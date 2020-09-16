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
import matplotlib.colors as mpc

from tools_LT import read_evar_only 

quick = True
#quick = False

def main( INFO, tlev=0, vname="QG", member=80, zlev_show=10, zlev_tgt=10, mem=0, ftit=None ):


    # read ens variables
    ew = read_evar_only( INFO, tlev=tlev, vname="W" )
#    eu = read_evar_only( INFO, tlev=tlev, vname="U" )
#    ev = read_evar_only( INFO, tlev=tlev, vname="V" )
    et = read_evar_only( INFO, tlev=tlev, vname="T" )
    eqh = read_evar_only( INFO, tlev=tlev, vname="QHYD" ) * 1.e3
    eqcrg = read_evar_only( INFO, tlev=tlev, vname="QCRG" )

    ew_m = ew[mem,:,:,:]
#    eu_m = eu[mem,:,:,:]
#    ev_m = ev[mem,:,:,:]
    et_m = et[mem,:,:,:]
    eqh_m = eqh[mem,:,:,:]
    eqcrg_m = eqcrg[mem,:,:,:]


    max_loc_w =  np.unravel_index(np.argmax( ew_m[zlev_tgt,:,:] ),
                                  ew_m[zlev_tgt,:,:].shape )
    cx_l = [ max_loc_w[1] ]
    cy_l = [ max_loc_w[0] ]
      
    for cy in cy_l:
       for cx in cx_l:
    

           ctime = INFO["time0"] + timedelta( seconds = int( tlev ) * INFO["DT"] )
           ft_sec =  int( (ctime - INFO["time00"] ).total_seconds() )
           ft_sec_a = int( tlev * INFO["DT"] )
       
           fig = plt.figure( figsize=(8.5, 8.5) ) # h:v
           #gs = gridspec.GridSpec(105, 155) # v:h
           gs = gridspec.GridSpec(105, 100) # v:h
       
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
           print( "ax1", vmax1,vmax1+dv,hmin1,hmin1+dh)
       
           hmin2 = hmin1_r + dh_r + 2*pdh
           hmin2_r = hmin2 + dh + pdh
           ax2 = plt.subplot(   gs[vmax1:vmax1+dv,hmin2:hmin2+dh] )
           ax2_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin2:hmin2+dh] )
           ax2_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin2_r:hmin2_r+dh_r] )
       
#           hmin3 = hmin2_r + dh_r + 2*pdh
#           hmin3_r = hmin3 + dh + pdh
#           ax3 = plt.subplot(   gs[vmax1:vmax1+dv,hmin3:hmin3+dh] )
#           ax3_t = plt.subplot( gs[vmax1_t:vmax1_t+dv_t,  hmin3:hmin3+dh] )
#           ax3_r = plt.subplot( gs[vmax1:vmax1+dv,  hmin3_r:hmin3_r+dh_r] )
       
           vmax3_t = vmax1 + dv + 2*pdv + pdv
           vmax3 = vmax3_t + dv_t + pdv
           ax3 = plt.subplot(   gs[vmax3:vmax3+dv,hmin1:hmin1+dh] )
           ax3_t = plt.subplot( gs[vmax3_t:vmax3_t+dv_t,    hmin1:hmin1+dh] )
           ax3_r = plt.subplot( gs[vmax3:vmax3+dv,  hmin1_r:hmin1_r+dh_r] )
       
           ax4 = plt.subplot(   gs[vmax3-dv_t-pdv:vmax3+dv,       hmin2:hmin2+dh] )
#           ax4_t = plt.subplot( gs[vmax3_t:vmax3_t+dv_t, hmin2:hmin2+dh] )
#           ax4_r = plt.subplot( gs[vmax3:vmax3+dv,       hmin2_r:hmin2_r+dh_r] )
       
#           ax6 = plt.subplot(   gs[vmax4:vmax4+dv,       hmin3:hmin3+dh] )
#           ax6_t = plt.subplot( gs[vmax4_t:vmax4_t+dv_t, hmin3:hmin3+dh] )
#           ax6_r = plt.subplot( gs[vmax4:vmax4+dv,       hmin3_r:hmin3_r+dh_r] )
       
       
           ax_l = [ ax1, ax2 , ax3, ax1_r, ax1_t, ax2_r, ax2_t, ax3_r, ax3_t, ]

 
           cmap_rb = plt.cm.get_cmap("RdBu_r")
           cmap_b = plt.cm.get_cmap("Blues_r")
           cmap_rb.set_over('gray', alpha=1.0)
           cmap_rb.set_under('gray', alpha=1.0)
           levs_rb = np.array([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
           levs_rb_w = np.arange( -60, 70, 10 )


           levs_rb_th = np.array([-5, -4, -3, -2, -1, -0.5, 1, 2, 3, 4, 5 ])

           levs_rb_tc = np.array([-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50])

           levs_b_tc = np.arange(-80, 10, 10)
           levs_b_tc = np.arange(-30, 35, 5)

           #levs_rb_qcrg = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, 
           #                        0.1, 0.2, 0.4, 0.6, 0.8, 1])
           levs_rb_qcrg = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.01,
                                   0.01, 0.05, 0.1, 0.2, 0.3, 0.4])

           levs_rb_qcrg = np.array([-0.6, -0.4, -0.2, -0.1, -0.05, -0.01,
                                   0.01, 0.05, 0.1, 0.2, 0.4, 0.6])

       
           colors1 = plt.cm.jet_r(np.linspace(0, 1, 128))
           colors2 = plt.cm.binary(np.linspace(0., 1, 128)) # w/k
           colors = np.vstack((colors1, colors2))
           cmap_tbb = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
           levs_tbb = np.arange(200,304,4)
       
           cmap_e = cmap_jet = plt.cm.get_cmap( "hot_r" )
           levs_e = np.arange( 0,100,4 )
           cmap_e.set_under('w', alpha=1.0)
       
       
           levs_dbz= np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
           #levs_qh= np.array([0.5, 1, 4, 6, 8, 10, 12, 14, 16])
           levs_qh= np.array([0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
           cmap_dbz = mcolors.ListedColormap(['cyan','dodgerblue',
                                              'lime', 'limegreen','yellow',
                                              'orange', 'red', 'firebrick', 'magenta',
                                              'purple'])
           cmap_dbz.set_under('w', alpha=1.0)
           cmap_dbz.set_over('gray', alpha=1.0)
       
           levs_vr = np.arange( -40, 44, 4 )
           cmap_vr = plt.cm.get_cmap( "Spectral_r" )
           cmap_vr.set_under('gray', alpha=1.0)
           cmap_vr.set_over('gray', alpha=1.0)
       
           # for GLM
           # simply accumulate
           ng = 4
           #ng = 2
           kernel = np.ones((ng,ng))         # accumulate
           #kernel = kernel / np.size(kernel) # average

           bbox = { 'facecolor':'w', 'alpha':0.95, 'pad':1.5, 'edgecolor':'w' }
       
       
           crs_l = [
                     "XY_s", "XY_s", "XY_s", #"XY_s", 
                     "YZ", "XZ", "YZ", "XZ", "YZ", "XZ",
                     "YZ", "XZ", "YZ", "XZ", "YZ", "XZ",
                   ]
       
           pnum_l = [ "(a)", 
                      "(b)",
                      "(c)",
                      "(d)",
                      "(e)",
                      ]

           VAR_l = [ 
                     ew_m[zlev_show,:,:], 
                     eqh_m[zlev_show,:,:], 
                     eqcrg_m[zlev_show,:,:], 
                     #eu_m[zlev_show,:,:], 
                     #ev_m[zlev_show,:,:], 
                     #et_m[0,:,:] - 273.15, # tc
                     np.transpose( ew_m[:,:,cx] ), 
                     ew_m[:,cy,:], 
                     np.transpose( eqh_m[:,:,cx] ), 
                     eqh_m[:,cy,:], 
                     np.transpose( eqcrg_m[:,:,cx] ), 
                     eqcrg_m[:,cy,:], 
                     #np.transpose( eu_m[:,:,cx] ), 
                     #eu_m[:,cy,:], 
                     #np.transpose( ev_m[:,:,cx] ), 
                     #ev_m[:,cy,:], 
                     #np.transpose( et_m[:,:,cx] ) - 273.15, 
                     #et_m[:,cy,:] - 273.15, 
#                       np.transpose( np.mean( ez[1:,:,:,cx], axis=0 ) ), # YZ
#                       np.mean( ez[1:,:,cy,:], axis=0 ),
  
                     "",
                     "",
                     "",
                     "",
                     "",
                     "",
                      ]
           levs_l = [ levs_rb_w, levs_qh, levs_rb_qcrg, 
                      levs_rb_w, levs_rb_w, levs_qh, levs_qh, levs_rb_qcrg, levs_rb_qcrg,
                      "", "", "", "", "", "",
                    ]
       
           cmap_l = [ cmap_rb, cmap_dbz, cmap_rb,  
                      cmap_rb, cmap_rb, cmap_dbz, cmap_dbz, cmap_rb, cmap_rb, 
                      "", "", "", "", "", "",
                    ]
           fig_tit =  "Ensemble mean"

           if mem is 0:
              tmem = "mean"
           else:
              tmem = str(mem).zfill(4)

           if ftit is not None:
              fig_tit = ftit

           ofig =  "4p_" + tmem + "_" + INFO["EXP"] + "_t" + str( ft_sec ).zfill(5) + "_ft" + str( ft_sec_a ).zfill(5) 
           odir = "png/4p_var_" + INFO["EXP"] + "_em"
          
           ###
       
           tit_l = [ 
                     "W",
                     "QHYD",
                     "QCRG", 
                     r'$T_c$', 
                     "Surface E", 
                     "Pseudo GLM", 
                     "", "", "", "", "", "",
                     "", "", "", "", "", "",
                   ]
           unit_l = [ 
                      '(m s$^{-1}$)', 
                      r'(g kg$^{-1}$)', 
                      r'(nCm$^{-3}$)',
                      '(K)', 
                      '(kV/m)', # E
                      '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r'min)', # GLM
                      '(dBZ)', 
                      '(dBZ)', 
                      '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r'min)',
                      '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r'min)',
                     ]
       

 

           xmin_l = 120
           xmax_l = 300
           ymin_l = 140
           ymax_l = 320
       
           xmin_s = xmin_l + 20
           xmax_s = xmax_l - 40 
           ymin_s = ymin_l + 20 
           ymax_s = ymax_l - 40 
       
           xmin_s = 140
           xmax_s = 250
           ymin_s = 160
           ymax_s = 250
       
           ymax_s = 260 - 20
           xmax_s = 260 - 20
           xmin_s = 120 + 20
           ymin_s = 120 + 20


           ctime = ( INFO["time0"] + timedelta(seconds=INFO["DT"] * tlev ) ).strftime('%H:%M:%S')
       
           for idx, ax in enumerate(ax_l):
        
              if VAR_l[idx] is "":
                 ax.axis("off")
                 continue
       
              if crs_l[idx] is "XY":
                 print(idx,tit_l[idx])
       
              xlabel = "X (km)"
              ylabel = "Y (km)"
       
              if crs_l[idx] is "XY":
                 xmin = xmin_l
                 xmax = xmax_l
                 ymin = xmin_l
                 ymax = ymax_l
                 xdgrid = 20
                 ydgrid = 20
       
              elif crs_l[idx] is "XY_s" or crs_l[idx] is "XY_skip":
                 xmin = xmin_s
                 xmax = xmax_s
                 ymin = ymin_s
                 ymax = ymax_s
                 xdgrid = 10
                 ydgrid = 10
       

              if crs_l[idx] is "XY_skip":
                 nskip = ng
       
                 kernel1d = np.ones(ng) / ng
                 xaxis = np.convolve( INFO["X"]*0.001, kernel1d, mode='same' )[nskip::nskip]
                 yaxis = np.convolve( INFO["X"]*0.001, kernel1d, mode='same' )[nskip::nskip]
              else:
                 nskip = 1
                 xaxis = INFO["X"][nskip::nskip] * 0.001
                 yaxis = INFO["Y"][nskip::nskip] * 0.001
       
              imin = nskip
              jmin = nskip

              if crs_l[idx] is "YZ":
                 xmin = 0.0
                 xmax = 15.0
                 xmax = 17.0

                 ymin = ymin_s
                 ymax = ymax_s
                 nskip = 1
                 imin = 0
                 jmin = 0
       
                 xdgrid = 2
                 ydgrid = 10
       
                 yaxis = INFO["Z"][:] * 0.001
                 xaxis = INFO["Y"][jmin::nskip] * 0.001
                 xlabel = "Z (km)"
       
       #          if ax is ax1_r:
       #             ax.vlines( x=INFO["Z"][zlev]*0.001, ymin=ymin, ymax=ymax,
       #                        colors="k",linestyles='dotted',linewidths=1.0 )

              elif crs_l[idx] is "XZ":
                 ymin = 0.0
                 ymax = 15.0
                 ymax = 17.0
                 xmin = xmin_s
                 xmax = xmax_s
       
                 nskip = 1
                 imin = 0
                 jmin = 0
       
                 ydgrid = 2
                 xdgrid = 10
       
                 xaxis = INFO["Z"][:] * 0.001
                 yaxis = INFO["X"][jmin::nskip] * 0.001
       
                 ylabel = "Z (km)"
       


              if crs_l[idx] is "XY_s":
                 ax.plot( INFO["X"][cx]*0.001, INFO["Y"][cy]*0.001, marker='s', alpha=1.0,
                           ms=8, markerfacecolor="None", markeredgecolor='k',
                           markeredgewidth=1.0, zorder=1 )
       
                 # Plot radar site
                 if ax is ax1 or ax is ax2:
                    ax.plot( 180.0, 180.0, marker='x', alpha=1.0,
                              ms=6, markerfacecolor="None", markeredgecolor='k',
                              markeredgewidth=1.0 )
 
              zlev_show_ = zlev_show
#              if ax is ax4 or ax is ax4_t or ax is ax4_r:
#                 zlev_show_ = 0
       
              if crs_l[idx] is "XY_s" or crs_l[idx] is "XY_skip":
                 ax.vlines( x=INFO["X"][cx]*0.001, ymin=ymin, ymax=ymax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
                 ax.hlines( y=INFO["Y"][cy]*0.001, xmin=xmin, xmax=xmax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
              if crs_l[idx] is "YZ":
                 ax.hlines( y=INFO["Y"][cy]*0.001, xmin=xmin, xmax=xmax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
                 ax.vlines( x=INFO["Z"][zlev_show_]*0.001, ymin=ymin, ymax=ymax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
       

                 ax.plot( INFO["X"][cx]*0.001, INFO["Z"][zlev_tgt]*0.001, #INFO["X"][cx]*0.001,
                          marker='s', alpha=1.0,
                          ms=18, markerfacecolor="b", markeredgecolor='k',
                          markeredgewidth=1.0, zorder=1 )
                 print("CHEC", INFO["Z"][zlev_tgt]*0.001)
       
                 ax.plot( INFO["Z"][zlev_tgt]*0.001, INFO["Y"][cy]*0.001,
                          marker='s', alpha=1.0,
                          ms=8, markerfacecolor="None", markeredgecolor='k',
                          markeredgewidth=1.0, zorder=1 )

              elif crs_l[idx] is "XZ":
                 ax.vlines( x=INFO["X"][cx]*0.001, ymin=ymin, ymax=ymax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
                 ax.hlines( y=INFO["Z"][zlev_show_]*0.001, xmin=xmin, xmax=xmax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
       
                 ax.plot( INFO["X"][cx]*0.001, INFO["Z"][zlev_tgt]*0.001,
                          marker='s', alpha=1.0,
                          ms=8, markerfacecolor="None", markeredgecolor='k',
                          markeredgewidth=1.0, zorder=1 )
       
       
              if crs_l[idx] is not "YZ":
                 ax.set_ylabel( ylabel, fontsize=6 )
       
              elif crs_l[idx] is not "XZ":
                 ax.set_xlabel( xlabel, fontsize=6 )
       
              x2d, y2d = np.meshgrid( yaxis, xaxis )
       
              print( "CHECK", idx, VAR_l[idx].shape, 
                     np.nanmax(VAR_l[idx]), np.nanmin(VAR_l[idx]) )
   
              cnorm = None    
              if ax is ax2 or ax is ax2_r or ax is ax2_t:
                 cnorm = mpc.BoundaryNorm(levs_l[idx], ncolors=cmap_l[idx].N, clip=False)
              SHADE = ax.contourf(x2d, y2d,
                                  VAR_l[idx][jmin::nskip,imin::nskip],
                                  levels=levs_l[idx],
                                  cmap=cmap_l[idx],
                                  norm=cnorm,
                                  extend='both',
                                  ) 
          
              ax.set_xlim( xmin, xmax )
              ax.set_ylim( ymin, ymax )
              ax.xaxis.set_ticks( np.arange(xmin, xmax, xdgrid) )
              ax.yaxis.set_ticks( np.arange(ymin, ymax, ydgrid) )
              ax.tick_params(axis='both', which='minor', labelsize=5 )
              ax.tick_params(axis='both', which='major', labelsize=5 )
     
#              if ax is ax3_t or ax is ax3_r:
#                 if ax is ax3_t:
#                    var2d = et_m[:,cy,:] - 273.15                 
#                 elif ax is ax3_r:
#                    var2d = np.transpose( et_m[:,:,cx] ) - 273.15                 
#
#                 CONT = ax.contour( x2d, y2d, 
#                                    var2d,
#                                    colors='b',
#                                    linewidths=0.2,
#                                    linestyles='solid',
#                                    levels=[ -40, -30, -20, -10, 0],
#                                   )
#                 ax.clabel( CONT, inline=True, fontsize=6, fmt=r'%2.0f$^\circ$C',  )

       #          ax.set_ylabel( ylabel, fontsize=6 )
       #
              ax.grid( axis='both', ls='dashed', lw=0.2 )
       
              tskip = 1
       
              if crs_l[idx] is not "XZ" and crs_l[idx] is not "YZ":
                 pos = ax.get_position()
                 cb_h = 0.01 #pos.height
                 cb_w = pos.width * 1.5
                 ax_cb = fig.add_axes( [pos.x0+0.0, pos.y0-0.055, cb_w, cb_h] )
                 cb = plt.colorbar( SHADE, cax=ax_cb, orientation = 'horizontal', 
                                    ticks=levs_l[idx][::tskip], extend='both' )
                 cb.ax.tick_params( labelsize=6 )
                 
                 ax.text( 1.0, -0.09, unit_l[idx],
                          fontsize=6, transform=ax.transAxes,
                          horizontalalignment='right',
                          verticalalignment='top', )
                 
          
              if crs_l[idx] is "XY":
                 rect = patches.Rectangle( (xmin_s,ymin_s), 
                                           xmax_s-xmin_s, ymax_s-ymin_s, 
                                           lw=1, edgecolor='r',facecolor='none' )
                 ax.add_patch(rect)
       
       
       #       if idx == 5:
       #          ax.tick_params( labelbottom=False ) 
       #       if idx == 4:
       #          ax.tick_params( labelleft=False ) 
       
              if crs_l[idx] is not "XZ" and crs_l[idx] is not "YZ":
                 ax.text(0.5, 0.95, tit_l[idx],
                         fontsize=10, transform=ax.transAxes,
                         horizontalalignment='center',
                         verticalalignment='top', 
                         bbox=bbox )

              if idx <= 2:
                 ax.text(0.1, 0.95, pnum_l[idx],
                         fontsize=10, transform=ax.transAxes,
                         horizontalalignment='center',
                         verticalalignment='top', 
                         bbox=bbox )
       

              
              if idx == 0:
                 fig.text(0.99, 0.96, 't = {0:.0f} min'.format( ft_sec/60.0 ), 
                         fontsize=11, #transform=ax.transAxes,
                         horizontalalignment='right',
                         verticalalignment='center')
       
       #       if idx == 2:
       #          ax.set_xticks(np.arange(0,300,2), minor=False)
       
           fig.suptitle( fig_tit, fontsize=18 )
       

           sumfp1d = np.zeros(INFO["ZDIM"])
       
           for tlev in range(1, 13):
              ctime = INFO["time0"] + timedelta(seconds=INFO["DT"]*tlev )
              # nature run
              # read variables
              #INFO["EXP"] = "2000m_NODA_0306"
              INFO["MEM"] = "mean"
              INFO["TYPE"] = "fcst"
              INFO["time0"] = datetime(2001, 1, 1, 1, 0)
              tlev_nat = int( ( ctime - datetime(2001, 1, 1, 1, 0) ).total_seconds() / INFO["DT"] )
              print( "DEBUG", tlev_nat, ctime)
              efp_nat_ = read_evar_only( INFO, tlev=tlev_nat, vname="FP" )
       
              efp_nat = efp_nat_[0,:,:,:]
              #efp_nat = np.where( efp_nat > 0.0, 1.0, 0.0)
       
              print("shape",efp_nat.shape, np.max(efp_nat), np.sum(efp_nat) )
              sumfp1d +=  np.sum(efp_nat , axis=(1,2) )
       
           print(sumfp1d)

           ax4.barh( INFO["Z"]*0.001, sumfp1d, color='k', height=0.4 )
       
           ax4.set_xlim(0, 70)
           ax4.set_ylim(0, 14)
           ax4.set_ylabel("Height (km)")
           ax4.set_xlabel("# of flashes")
       
           ax4.text( 0.5, 0.95, "Flash count\n(FT=60-120min)",
                     fontsize=10, transform=ax4.transAxes,
                     horizontalalignment='center',
                     verticalalignment='top', )
      
           ax4.text( 0.1, 0.95, "(d)",
                     fontsize=10, transform=ax4.transAxes,
                     horizontalalignment='center',
                     verticalalignment='top', 
                     bbox=bbox )
      
       
           print( ofig, odir )
        
           if not quick:
              os.makedirs(odir, exist_ok=True)
              plt.savefig(os.path.join(odir,ofig),
                          bbox_inches="tight", pad_inches = 0.1)
              plt.cla()
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
EXP = "2000m_NODA_0306"
EXP = "2000m_NODA_0601"

EXP = "2000m_NODA_0723"

member = 320
member = 0

time0 = datetime( 2001, 1, 1, 1, 20, 0 ) 
time0 = datetime( 2001, 1, 1, 1, 30, 0 ) 
time0 = datetime( 2001, 1, 1, 1, 0, 0 ) 

GTOP = "/data_honda01/honda/SCALE-LETKF/scale-LT/OUTPUT"
TYPE = "fcst"
time00 = datetime( 2001, 1, 1, 0, 0, 0 )

INFO = {"XDIM":XDIM, "YDIM":YDIM, "NBAND":10, "TDIM":TDIM,
        "X":X, "Y":Y , "BAND":BAND, "T":T, "GTOP":GTOP,
        "ZDIM":ZDIM, "Z":Z, "DT":DT,
        "TYPE":TYPE, "EXP":EXP,
        "time0": time0, "time00": time00  }

tmin = 1
#tmin = 2
#tmin = 3
#tmax = tmin + 1
tmax = 13
vname = "QG"
#vname = "CS"
vname = "QS"
#vname = "CG"
#vname = "QCRG"
#vname = "QHYD"
#vname = "W"
#vname = "QG"

vname_l = [ "QR", "QI", "QV", "CR", "CI", 
            "QS", "QG", "CG", "CS", "QHYD", "QCRG", "W"]

vname_l = [ "QHYD", "QCRG","W",
            ]

#vname_l = [ "QCRG" ] # DEBUG

zlev_show = 10
zlev_tgt = 10


mem = 0 # member index, 0: mean


ftit = "Nature run"

for vname in vname_l:
    for tlev in range( tmin, tmax ):
        main( INFO, tlev=tlev, vname=vname, member=member, zlev_show=zlev_show, zlev_tgt=zlev_tgt, mem=mem, ftit=ftit )

