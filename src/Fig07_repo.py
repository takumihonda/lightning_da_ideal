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

from tools_LT import read_evar4d_nc, read_evars, get_ecor, get_eGLM, band2wavelength

quick = True
#quick = False


def main( INFO, tlev=0, vname="QG", cx_l=[100], cy_l=[100], member=80, COR=True, zlev_show=10, zlev_tgt=10, mmem=0, fp_acum=1, 
          vname1="QG", vname2="QS", vname3="T", band=13 ):

    data_path = "../../dat4figs/Fig07"
    os.makedirs( data_path, exist_ok=True )

    cx = cx_l[0]
    cy = cy_l[0]

    mmem_ = mmem
    if mmem <= 0:
       mmem_ = 0

    # read obs variables & ens variables
#    etbb, ez, efp, evar1 = read_evars( INFO, tlev=tlev, vname=vname1, member=member )

    ac1 = 0
    ac2 = 1
    ac3 = 5

#    # debug
#    if quick:
#       ac2 = 0
#       ac3 = 0
#       fp_acum = 1

#    _, _, _, evar2 = read_evars( INFO, tlev=tlev, vname=vname2, member=member )
#    _, _, _, evar3 = read_evars( INFO, tlev=tlev, vname=vname3, member=member )

#    for dt_ in range( 1, fp_acum ):
#       _,  _, efp_, _ = read_evars( INFO, tlev=tlev-dt_, vname=vname, member=member )
#       efp += efp_



    if zlev_show < 0:
       zlev_show = 10
    if zlev_tgt < 0:
       zlev_tgt = 10

    print("zlev",zlev_tgt, zlev_show)
    print("")



    pnum_l = [ "(a)", 
               "(b)",
               "(c)",
               "(d)",
               "(e)",
               "(f)",
               ]




#    mcnt2d= np.sum( np.where( np.sum(efp[1:,:,:,:], axis=1) > 0.0, 1.0, 0.0 ) , axis=0 )
#       
#    cvar_mean = np.mean( ez[1:,:,:,:], axis=0 )

    for cy in cy_l:
       for cx in cx_l:
 
#           print( mcnt2d[cy,cx])
#           if mcnt2d[cy,cx] < mmem:
#              continue
#   
#           print("\nGRID:",cx, cy,"\n")
#
#           print("fp each levs", np.mean(efp[1:,:,cy,cx], axis=0) )
#           print("")
#           print("fp each mems", np.sum(efp[1:,:,cy,cx], axis=1) )
#           mcnt = np.sum( np.where( np.sum(efp[1:,:,cy,cx], axis=1) > 0.0, 1.0, 0.0 ) )
#           print("Flash mem:", mcnt)
#           print("")

#           print(INFO["X"][cx]*0.001, INFO["Y"][cy]*0.001)
#           continue

           ctime = INFO["time0"] + timedelta( seconds = int( tlev ) * INFO["DT"] )
           ft_sec =  int( (ctime - INFO["time00"] ).total_seconds() )
           ft_sec_a = int( tlev * INFO["DT"] )
       
       
       
 
           dv = 8
           dh = 8      
           dv2 = 4
           dh2 = 2

           #v_tot = dv2 + 1 + dv + 2
           #h_tot = dh + 2 + dh + 2 + dh 
           v_tot = dv2 + 1 + dv + 2 + dv2 + 1 + dv + 2
           h_tot = dh + 2 + dh + 2 + dh 

           hsize = 8.0
           vsize = hsize * v_tot / h_tot
           fig = plt.figure( figsize=(hsize, vsize) )

           gs = gridspec.GridSpec( 8, 5, 
                                   height_ratios=( dv2, 1, dv, 2, dv2, 1, dv, 2 ), 
                                   width_ratios=( dh, 2, dh, 2, dh ) )
           axs = [ 
                   plt.subplot(gs[0, 0]), 
                   plt.subplot(gs[2, 0]),
       
                   plt.subplot(gs[0, 2]), 
                   plt.subplot(gs[2, 2]), 
       
                   plt.subplot(gs[0, 4]), 
                   plt.subplot(gs[2, 4]),

                   plt.subplot(gs[4, 0]), 
                   plt.subplot(gs[6, 0]),
       
                   plt.subplot(gs[4, 2]), 
                   plt.subplot(gs[6, 2]), 
       
                   plt.subplot(gs[4, 4]), 
                   plt.subplot(gs[6, 4]),
                 ]

           ax1_t = axs[0]
           ax1 = axs[1]
           ax2_t = axs[2]
           ax2 = axs[3]
           ax3_t = axs[4]
           ax3 = axs[5]

           ax4_t = axs[6]
           ax4 = axs[7]

           ax5_t = axs[8]
           ax5 = axs[9]

           ax6_t = axs[10]
           ax6 = axs[11]

           fig.subplots_adjust( left=0.06, right=0.94, top=0.94, bottom=0.06,
                                wspace=0.0, hspace=0.0, 
                              )
       

      

           ax_l = [ ax1, ax2, ax3,  ax4, ax5, ax6, 
                    ax1_t, ax2_t, ax3_t,
                    ax4_t, ax5_t, ax6_t,
                    ] #ax5_r, ax5_t, ax6_r, ax6_t ]
 
           cmap_rb = plt.cm.get_cmap("RdBu_r")
           cmap_rb.set_over('gray', alpha=1.0)
           cmap_rb.set_under('gray', alpha=1.0)
           levs_rb = np.arange( -0.8, 0.9, 0.1 )
           levs_rb = np.array([-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8])
           levs_rb = np.array([-0.8, -0.6, -0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
       #    levs_rb = np.arange( -0.6, 0.7, 0.1 )
       
           colors1 = plt.cm.jet_r(np.linspace(0, 1, 128))
           colors2 = plt.cm.binary(np.linspace(0., 1, 128)) # w/k
           colors = np.vstack((colors1, colors2))
           cmap_tbb = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
           levs_tbb = np.arange(200,304,4)
       
           cmap_e = cmap_jet = plt.cm.get_cmap( "hot_r" )
           levs_e = np.arange( 0,100,4 )
           cmap_e.set_under('w', alpha=1.0)
       
           cmap_fp = cmap_jet = plt.cm.get_cmap( "hot_r" )
           #levs_fp = np.arange( 0,3.5,0.5 )
           levs_fp = np.arange( 0, 420, 20 )
           levs_fp_small = np.arange( 0, 0.5, 0.05 )
           levs_fp_small = levs_fp 
           #levs_fp = np.arange( 0,55,5 )
           cmap_fp.set_under('w', alpha=1.0)
       
           levs_dbz= np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
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

           bbox = { 'facecolor':'w', 'alpha':1.0, 'pad':1.5, 'edgecolor':'w' }
       
       
       
#           eglm = get_eGLM( efp, kernel )
#           var1d_glm = eglm[:,cy,cx]
#
#           ecor_glm1 = get_ecor( var1d_glm, evar1 )
#           ecor_glm2 = get_ecor( var1d_glm, evar2 )
#           ecor_glm3 = get_ecor( var1d_glm, evar3 )
       
#           print( etbb.shape )
#           var1d_tbb = etbb[:,band-7,cy,cx]
#           print( "debug ", np.max( np.abs( etbb[10,9-7,:,:] - etbb[10,13-7,:,:] ) ))
#           ecor_tbb1 = get_ecor( var1d_tbb, evar1 )
#           ecor_tbb2 = get_ecor( var1d_tbb, evar2 )
#           ecor_tbb3 = get_ecor( var1d_tbb, evar3 )
       

           crs_l = [
                     "XY_s", "XY_s", "XY_s",  "XY_s", "XY_s", "XY_s",
                     "XZ", "XZ", "XZ", "XZ", "XZ", "XZ",
                  #   "YZ", "XZ", #"YZ", "XZ", "YZ", "XZ",
                   ]
       
  
           if COR:
          
#              VAR_l = [ 
#                        ecor_glm1[zlev_show,:,:],    # GLM
#                        ecor_glm2[zlev_show,:,:],    # GLM
#                        ecor_glm3[zlev_show,:,:],    # GLM
#                        ecor_tbb1[zlev_show,:,:],    # TBB
#                        ecor_tbb2[zlev_show,:,:],    # TBB
#                        ecor_tbb3[zlev_show,:,:],    # TBB
#          
#                        ecor_glm1[:,cy,:], # XZ
#                        ecor_glm2[:,cy,:], # XZ
#                        ecor_glm3[:,cy,:], # XZ
#                        ecor_tbb1[:,cy,:], # XZ
#                        ecor_tbb2[:,cy,:], # XZ
#                        ecor_tbb3[:,cy,:], # XZ
#                      ]

              cmap_l = [ cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, 
                         cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, 
                         cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, 
                         cmap_rb, cmap_rb, cmap_rb, cmap_rb, cmap_rb, 
                       ]
              levs_l = [ levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, 
                         levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, 
                         levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, 
                         levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, levs_rb, ]
       
              fig_tit =  "Ensemble-based correlations" 
       
              ofig =  "6p_acm_var_{0:}_{1:}_{2:}_{3:}_obs_init{4:}_t{5:}_ft{6:}_x{7:}_y{8:}_zs{9:}_zt{10:}_mmem{11:}_fpacum{10:}".format( INFO["EXP"], vname1, vname2, INFO["time0"].strftime('%H%M'), \
                       str( ft_sec ).zfill(5), str( ft_sec_a ).zfill(5),  str(cx).zfill(3), str(cy).zfill(3),  str(zlev_show).zfill(2), str(zlev_tgt).zfill(2), str( mmem_ ).zfill(4), str(fp_acum).zfill(2) ) + '.pdf'
              #odir = "png/fig0624/6p_acm_var_" + INFO["EXP"] + "/" + str( ft_sec ).zfill(5) + "_ft" + str( ft_sec_a ).zfill(5) 
              odir = "pdf/fig20210624/6p_acm_var_" + INFO["EXP"] + "/" + str( ft_sec ).zfill(5) + "_ft" + str( ft_sec_a ).zfill(5) 
       
           else:
              VAR_l = [ 
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                      ]
              levs_l = [ levs_dbz, levs_vr, levs_fp_small, levs_tbb, levs_e, levs_fp, 
                         levs_dbz, levs_dbz, levs_vr, levs_vr, levs_fp_small, levs_fp_small, 
                         "", "", "", "", "", "",
                       ]
          
              cmap_l = [ cmap_dbz, cmap_vr, cmap_fp, cmap_tbb, cmap_e, cmap_fp, 
                         cmap_dbz, cmap_dbz, cmap_vr, cmap_vr, cmap_fp, cmap_fp, 
                         "", "", "", "", "", "",
                       ]
              crs_l[5] = "XY_skip"
              fig_tit =  "Ensemble mean"
              ofig =  "9p_emean_" + INFO["EXP"] + "_obs_t" + str( ft_sec ).zfill(5) + "_ft" + str( ft_sec_a ).zfill(5) + "_B{0:0=2}".format( band ) + '.pdf'
              #odir = "png/fig0624/9p_obs_" + INFO["EXP"] + "_em"
              odir = "pdf/fig20210624/9p_obs_" + INFO["EXP"] + "_em"
       
          
           ###
       
           tit_l = [ 
                     "GLM & {0:}".format( vname1 ), 
                     "GLM & {0:}".format( vname2 ), 
                     "GLM & {0:}".format( vname3 ), 
                     r'IR ({0:.1f} $\mu$m) & {1:}'.format( band2wavelength( band=band ), vname1 ), 
                     r'IR ({0:.1f} $\mu$m) & {1:}'.format( band2wavelength( band=band ), vname2 ), 
                     r'IR ({0:.1f} $\mu$m) & {1:}'.format( band2wavelength( band=band ), vname3 ), 
                     "", "", "", "", "", "",
                     "", "", "", "", "", "",
                   ]
           unit_l = [ 
                      '(dBZ)', 
                      '(K)', 
                      '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r' min)', # GLM
                      '(dBZ)', 
                      '(dBZ)', 
                      '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r' min)',
                      '(flash/' + str( int( INFO["DT"]/60.0 ) ) + r' min)',
                     ]
       

 

           xmin_l = 120
           xmax_l = 300
           ymin_l = 120 + 10
           ymax_l = 320
       

           xmin_s = 150
           ymin_s = 170

           dx = 100
           dy = dx * dv / dh
    
           xmax_s = xmin_s + dx
           ymax_s = ymin_s + dy

           ctime = ( INFO["time0"] + timedelta(seconds=INFO["DT"] * tlev ) ).strftime('%H:%M:%S')
       
           for idx, ax in enumerate(ax_l):
        
#              if VAR_l[idx] is "":
#                 ax.axis("off")
#                 continue
       
              if crs_l[idx] is "XY":
                 print(idx,tit_l[idx])
       
              fn = '{0:}/data{1:0=2}.npz'.format( data_path, idx )
              print( fn )

              xlabel = "X (km)"
              ylabel = "Y (km)"
       
              if idx <= 5:
                 ax.text(0.1, 0.95, pnum_l[idx],
                         fontsize=10, transform=ax.transAxes,
                         horizontalalignment='center',
                         verticalalignment='top', 
                         zorder=5,
                         bbox=bbox )


              if crs_l[idx] is "XY":
                 xmin = xmin_l
                 xmax = xmax_l
                 ymin = ymin_l
                 ymax = ymax_l
                 xdgrid = 20
                 ydgrid = 20
       
              elif crs_l[idx] is "XY_s" or "XY_skip":
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
       

              elif crs_l[idx] is "XZ":
                 ymin = 0.0
                 ymax = 15.0
                 #ymax = 17.0
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
                           markeredgewidth=1.0 )
       
                 # Plot radar site
                 if ax is ax1:
                    ax.plot( 180.0, 180.0, marker='x', alpha=1.0,
                              ms=6, markerfacecolor="None", markeredgecolor='k',
                              markeredgewidth=1.0 )
       
       
              if crs_l[idx] is "XY_s" or crs_l[idx] is "XY_skip":
                 ax.vlines( x=INFO["X"][cx]*0.001, ymin=ymin, ymax=ymax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
                 ax.hlines( y=INFO["Y"][cy]*0.001, xmin=xmin, xmax=xmax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
              if crs_l[idx] is "YZ":
                 ax.hlines( y=INFO["Y"][cy]*0.001, xmin=xmin, xmax=xmax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
                 ax.vlines( x=INFO["Z"][zlev_show]*0.001, ymin=ymin, ymax=ymax,
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
                 ax.hlines( y=INFO["Z"][zlev_show]*0.001, xmin=xmin, xmax=xmax,
                            colors="k",linestyles='dotted',linewidths=1.0 )
       
                 ax.plot( INFO["X"][cx]*0.001, INFO["Z"][zlev_tgt]*0.001,
                          marker='s', alpha=1.0,
                          ms=8, markerfacecolor="None", markeredgecolor='k',
                          markeredgewidth=1.0, zorder=1 )
       
       
              if crs_l[idx] is "XZ":
                 ax.set_ylabel( ylabel, fontsize=7 )
       
              if crs_l[idx] is not "XZ":
                 ax.set_xlabel( xlabel, fontsize=7 )
                 ax.set_ylabel( ylabel, fontsize=7 )
       
              x2d, y2d = np.meshgrid( yaxis, xaxis )
       
#              print( "CHECK", idx, VAR_l[idx].shape, 
#                     np.nanmax(VAR_l[idx]), np.nanmin(VAR_l[idx]) )
       
#              # ensemble-mean z
#              if crs_l[idx] is "XY_s":
#                 cvar = cvar_mean[zlev_show,jmin::nskip,imin::nskip]
#              elif crs_l[idx] is "XZ":
#                 cvar = cvar_mean[:,cy,imin::nskip]
#              elif crs_l[idx] is "YZ":
#                 cvar = np.transpose( cvar_mean[:,jmin::nskip,cx] )

#              np.savez( fn, data=VAR_l[idx][jmin::nskip,imin::nskip], cdata=cvar )
              cdata = np.load( fn )['cdata']

              CONT = ax.contour(x2d, y2d,
                                cdata, 
#                                cvar, 
                                levels=np.arange(30,110,10),
                                linewidths=0.5,
                                linestyles="solid",
                                colors='k',
                                zorder=4,
                                ) 
              ax.clabel( CONT, CONT.levels, inline=True, inline_spacing=-5, 
                  fontsize=6, fmt='%2.0fdBZ',  )

#              if COR and ( crs_l[idx] is "XY_s" or \
#                 crs_l[idx] is "XZ" or \
#                 crs_l[idx] is "YZ" ):

              data = np.load( fn )['data']
       
              SHADE = ax.contourf(x2d, y2d,
                                  data,
#                                  VAR_l[idx][jmin::nskip,imin::nskip],
                                  levels=levs_l[idx],
                                  cmap=cmap_l[idx],
                                  extend='both',
                                  ) 
          
              ax.set_xlim( xmin, xmax )
              ax.set_ylim( ymin, ymax )
              ax.xaxis.set_ticks( np.arange(xmin, xmax, xdgrid) )
              ax.yaxis.set_ticks( np.arange(ymin, ymax, ydgrid) )
              ax.tick_params(axis='both', which='minor', labelsize=7 )
              ax.tick_params(axis='both', which='major', labelsize=7 )
     

       
       #          ax.set_ylabel( ylabel, fontsize=6 )
       #
              ax.grid( axis='both', ls='dashed', lw=0.2 )
       
              if COR:
                 tskip = 1
              else:
                 tskip = 2
       
       
              #if crs_l[idx] is not "XZ" and crs_l[idx] is not "YZ":
              if ax is ax5:
                 pos = ax.get_position()
                 cb_h = 0.02 # pos.height*1.5
                 cb_w = pos.width * 2.5
                 ax_cb = fig.add_axes( [ pos.x0-cb_w*0.5, pos.y0-0.05-cb_h, 
                                         cb_w, cb_h] )
                 cb = plt.colorbar( SHADE, cax=ax_cb, orientation='horizontal', 
                                    ticks=levs_l[idx][::tskip], extend='both' )
                 cb.ax.tick_params( labelsize=7 )
                 
          
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
              
              if idx == 0:
                 fig.text(0.95, 0.05, "t = {0:.0f} min\n(FT={1:.0f} min)".format( ft_sec/60, ft_sec_a/60.0 ),
                         fontsize=10,
                         ha='right', va='bottom',
                         )
       
       #       if idx == 2:
       #          ax.set_xticks(np.arange(0,300,2), minor=False)
       
           fig.suptitle( fig_tit, fontsize=12 )
       
       
           print( ofig )
        
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

tmax = 13
tmin = 2  # 600
tmin = 6  # 1800
#tmin = 3
tmax = tmin + 1

vname_l = [ "QR", "QI", "QV", "CR", "CI", 
            "QS", "QG", "CG", "CS", "QHYD", "QCRG", "W"]

vname_l = [ "QHYD", "QCRG","QG",
            ]

#vname_l = [ "QS", "QG", ]

#vname_l = [ "QCRG" ] # DEBUG

zlev_show = 13
zlev_tgt = 13

COR = False
COR = True


cx_l = np.arange( 70, 105, 5)
cy_l = np.arange( 70, 105, 5)

cx_l = np.arange( 60, 120, 4)
cy_l = np.arange( 75, 120, 4)

#cx_l = np.arange( 72, 120, 4)
#cy_l = [95]

# Fig 3
#vname_l = [ "QHYD" ]
##vname_l = [ "QG" ]
#cx_l = [ 100 ]
#cy_l = [ 111 ]

# Fig S3
#vname_l = [ "QHYD" ]
#cx_l = [ 84 ]
#cy_l = [ 95 ]



cx_l = np.arange( 80, 120, 4)
cy_l = np.arange( 92, 120, 4)

cx_l = np.arange( 96, 104, 1)
cy_l = np.arange( 96, 104, 1)

cx_l = [ 103 ]
cy_l =  [ 110 ]



cx_l = np.arange( 70, 124, 4)
cy_l = np.arange( 90, 124, 4)


#cx_l = np.arange( 80, 122, 2)
#cy_l = np.arange( 88, 122, 2)
#

cx_l = np.arange( 82, 122, 1 )
cy_l = np.arange( 88, 122, 1 )

cx_l = [ 95 ]
cy_l =  [ 101 ]

cx_l = [ 93, 94, 95, 96, 97 ]
cy_l =  [ 99, 100, 101, 102, 103 ]


cx_l = np.arange( 82, 122, 2 )
cy_l = np.arange( 88, 122, 2 )

vname_l = [ "U", "V", "P", "T", "NG" ]
vname_l = [ "QHYD" ]
#Fig 4
cx_l = [ 97 ]
cy_l =  [ 101, ]

#vname_l = [ "QG" ]
#vname_l = [ "W" ]


#vname_l = [ "EABS" ]
#print(INFO["Z"][zlev_show])
#sys.exit()


#vname_l = [ "T" ]

vname1 = "W"
vname2 = "QV"
vname3 = "T"

mmem = 1
#mmem = 32

fp_acum = 2
fp_acum = 6
#fp_acum = 1 # DEBUG

if fp_acum > tmin:
   fp_acum = tmin

zlev_tgts = 21
#zlev_tgts = 14
#zlev_tgts = 11

band = 9

zlev_tgts = 14
zlev_tgte = zlev_tgts+1
for zlev_tgt in range(zlev_tgts, zlev_tgte, 1):
   zlev_show = zlev_tgt

   for tlev in range( tmin, tmax ):
       main( INFO, tlev=tlev, vname1=vname1,
             vname2=vname2, vname3=vname3,
             cx_l=cx_l, cy_l=cy_l, COR=COR, member=320, zlev_show=zlev_show, zlev_tgt=zlev_tgt, mmem=mmem, fp_acum=fp_acum, band=band )

   if not COR:
      sys.exit()

#   sys.exit()
