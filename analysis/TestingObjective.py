#!/usr/bin/env python
# -*- coding: utf-8 -*-   

# This file tests the training results with the optimization function

from __future__ import print_function
import sys
import h5py
import os
import numpy as np
import glob
import numpy.core.umath_tests as umath
import time
import math
import ROOT

import setGPU # if caltech
import GANutilsANG_concat as ang

def main():
    # All of the following needs to be adjusted
    from EcalCondGanAngle_3d_1loss import generator # architecture
    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*VarAngleMeas_*.h5" # path to data
    genpath = "3d_angleweights_1loss/params_generator*.hdf5"# path to weights
    sorted_path = 'Anglesorted'  # where sorted data is to be placed
    plotsdir = 'angle_optimization_2loss_pos' # plot directory
    particle = "Ele" 
    scale = 2
    threshold = 1e-4
    g= generator(latent_size=256)
    start = 10
    gen_weights=[]
    disc_weights=[]
    ang.safe_mkdir(plotsdir)
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    #gen_weights=gen_weights[:2]
    result = GetResults(metric, plotsdir, gen_weights, g, datapath, sorted_path, particle, scale, thresh=threshold)
    PlotResultsRoot(result, plotsdir, start)
   
#Plots results in a root file
def PlotResultsRoot(result, resultdir, start):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    c1.SetGrid ()
    legend = ROOT.TLegend(.6, .6, .9, .9)
    mg=ROOT.TMultiGraph()
    color1 = 2
    color2 = 8
    color3 = 4
    num = len(result)
    print(num)
    total = np.zeros((num))
    energy_e = np.zeros((num))
    pos_e = np.zeros((num))
    epoch = np.zeros((num))
      
    mine = 100
    minp = 100
    mint = 100
    for i, item in enumerate(result):
      epoch[i] = i  
      total[i]=item[0]
      if item[0]< mint:
         mint = item[0]
         mint_n = i
      pos_e[i]=item[1]
      if item[1]< minp:
         minp = item[1]
         minp_n = i
      energy_e[i]=item[2]
      if item[2]< mine:
         mine = item[2]
         mine_n = i
    gt  = ROOT.TGraph( num- start , epoch[start:], total[start:] )
    gt.SetLineColor(color1)
    mg.Add(gt)
    legend.AddEntry(gt, "Total error min = {} (epoch {})".format(mint, mint_n), "l")
    ge = ROOT.TGraph( num- start , epoch[start:], energy_e[start:] )
    ge.SetLineColor(color2)
    legend.AddEntry(ge, "Energy error min = {} (epoch {})".format(mine, mine_n), "l")
    mg.Add(ge)
    gp = ROOT.TGraph( num- start , epoch[start:], pos_e[start:])
    gp.SetLineColor(color3)
    mg.Add(gp)
    legend.AddEntry(gp, "Position error  = {} (epoch {})".format(minp, minp_n), "l")
    c1.Update()
    mg.SetTitle("Optimization function: Mean Relative Error on position and energy;Epochs;Error")
    mg.Draw('ALP')
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result.pdf"))
      
    gt.Fit("pol1")
    gt.GetFunction("pol1").SetLineColor(color1)
    gt.GetFunction("pol1").SetLineStyle(2)

    ge.Fit("pol1")
    ge.GetFunction("pol1").SetLineColor(color2)
    ge.GetFunction("pol1").SetLineStyle(2)
        
    gp.Fit("pol1")
    gp.GetFunction("pol1").SetLineColor(color3)
    gp.GetFunction("pol1").SetLineStyle(2)
        
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_Linfit.pdf"))
    gt.Fit("pol2")
    gt.GetFunction("pol2").SetLineColor(color1)
    gt.GetFunction("pol2").SetLineStyle(2)
        
    ge.Fit("pol2")
    ge.GetFunction("pol2").SetLineColor(color2)
    ge.GetFunction("pol2").SetLineStyle(2)
        
    gp.Fit("pol2")
    gp.GetFunction("pol2").SetLineColor(color3)
    gp.GetFunction("pol2").SetLineStyle(2)
        
    c1.Update()
    c1.Print(os.path.join(resultdir, "pol2fit.pdf"))
    print ('The plot is saved to {}'.format(resultdir))

# results are obtained using metric and saved to a log file
def GetResults(metric, resultdir, gen_weights, g, datapath, sorted_path, particle="Ele", scale=100, thresh=1e-6):
    resultfile = os.path.join(resultdir,  'result_log.txt')
    file = open(resultfile,'w')
    result = []
    for i in range(len(gen_weights)):
       if i==0:
         result.append(analyse(g, False,True, gen_weights[i], datapath, sorted_path, metric, scale, particle, thresh=thresh)) # For the first time when sorted data is not saved we can make use opposite flags
       else:
         result.append(analyse(g, True, False, gen_weights[i], datapath, sorted_path, metric, scale, particle, thresh=thresh))
       file.write("{}\t{}\t{}\n".format(result[i][0], result[i][1], result[i][2]))
    #print all results together at end                                                                               
    for i in range(len(gen_weights)):                                                                                            
       print ('The results for ......',gen_weights[i])
       print (" The result for {} = {:.4f} , {:.4f}, {:.4f}".format(i, result[i][0], result[i][1], result[i][2]))
    file.close
    print ('The results are saved to {}.txt'.format(resultfile))
    return result

# If reduced data is to be used in analyse function the line:
#   var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
# has to be replaced with:
#   var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, Data= GetAngleData_reduced, thresh=thresh)

def GetAngleData_reduced(datafile, thresh=1e-6):
    #get data for training
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    eta = np.array(f.get('eta')) + 0.6
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, eta, ecal
                              
# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(g, read_data, save_data, gen_weights, datapath, sorted_path, optimizer, xscale=100, particle="Ele", thresh=1e-6):
   print ("Started")
   num_events=2000
   num_data = 140000
   ascale = 1
   Test = True
   latent= 256
   m = 2
   var = {}
   energies = [110, 150, 190]
   sorted_path= sorted_path 
   #g =generator(latent)
   if read_data:
     start = time.time()
     var = ang.load_sorted(sorted_path + "/*.h5", energies)
     sort_time = time.time()- start
     print ("Events were loaded in {} seconds".format(sort_time))
   else:
     Trainfiles, Testfiles = ang.DivideFiles(datapath, nEvents=num_data, EventsperFile = 5000, Fractions=[.9,.1], datasetnames=["ECAL"], Particles =[particle])
     if Test:
       data_files = Testfiles
     else:
       data_files = Trainfiles + Testfiles
     start = time.time()
     #energies = [50, 100, 200, 250, 300, 400, 500]
     var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
     data_time = time.time() - start
     print ("{} events were loaded in {} seconds".format(num_data, data_time))
     if save_data:
        ang.save_sorted(var, energies, sorted_path)        
   total = 0
   for energy in energies:
     var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
     total += var["index" + str(energy)]
     data_time = time.time() - start
   print ("{} events were put in {} bins".format(total, len(energies)))
   g.load_weights(gen_weights)
              
   start = time.time()
   for energy in energies:
     var["events_gan" + str(energy)] = ang.generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100, var["angle" + str(energy)] * ascale, latent )
     var["events_gan" + str(energy)] = var["events_gan" + str(energy)]/xscale
   gen_time = time.time() - start
   print ("{} events were generated in {} seconds".format(total, gen_time))
   calc={}
   print("Weights are loaded in {}".format(gen_weights))
   for energy in energies:
     x = var["events_act" + str(energy)].shape[1]
     y = var["events_act" + str(energy)].shape[2]
     z = var["events_act" + str(energy)].shape[3]
     var["ecal_act"+ str(energy)] = np.sum(var["events_act" + str(energy)], axis = (1, 2, 3))
     var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
     calc["sumsx_act"+ str(energy)], calc["sumsy_act"+ str(energy)], calc["sumsz_act"+ str(energy)] = ang.get_sums(var["events_act" + str(energy)])
     calc["sumsx_gan"+ str(energy)], calc["sumsy_gan"+ str(energy)], calc["sumsz_gan"+ str(energy)] = ang.get_sums(var["events_gan" + str(energy)])
     calc["momentX_act" + str(energy)], calc["momentY_act" + str(energy)], calc["momentZ_act" + str(energy)]= ang.get_moments(calc["sumsx_act"+ str(energy)], calc["sumsy_act"+ str(energy)], calc["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
     calc["momentX_gan" + str(energy)], calc["momentY_gan" + str(energy)], calc["momentZ_gan" + str(energy)] = ang.get_moments(calc["sumsx_gan"+ str(energy)], calc["sumsy_gan"+ str(energy)], calc["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
   return optimizer(calc, energies, m, x, y, z)                                        
 
def metric(var, energies, m, x=25, y=25, z=25):
   metricp = 0
   metrice = 0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     x_act= np.mean(var["momentX_act"+ str(energy)], axis=0)
     x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
     x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
     y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
     y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
     z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
     z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
     var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
     var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
     var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
     #Taking absolute of errors and adding for each axis then scaling by 3
     var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
     #Summing over moments and dividing for number of moments
     var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
     metricp += var["pos_total"+ str(energy)]
     #Take profile along each axis and find mean along events
     sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis=0), np.mean(var["sumsz_act" + str(energy)], axis=0)
     sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
     var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
     var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
     var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
     #Take absolute of error and mean for all events
     var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/x
     var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/y
     var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z
     
     var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
     metrice += var["eprofile_total"+ str(energy)]
   metricp = metricp/len(energies)
   metrice = metrice/len(energies)    
   tot = metricp + metrice
   
   print(" Total Position Error = %.4f\t Total Energy Profile Error =   %.4f" %(metricp, metrice))
   print(" Total Error =  %.4f" %(tot))
   return(tot, metricp, metrice)

if __name__ == "__main__":
   main()
