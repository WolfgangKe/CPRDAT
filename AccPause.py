#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:10:15 2021

@author: kernwo
"""

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt


#class AccPause:
    
#### In Klasse speichern und importieren 

def av_mean(k,data): # Calculates a centered average mean of data over k elements. 
                    # data is zero-padded at the boundaries
    n=len(data)
    if k%2==1: #Calculate centered part of data
        erg=np.zeros(n-k+1) 
        for i in range(k):
            erg+=data[i:n-k+i+1]
        erg=erg/k
    else:
        erg=np.zeros(n-k)
        for i in range(k):
            erg+=data[i:n-k+i]
        erg=erg/k      
    erginit=np.zeros(k//2) # initial part
    ergfinal=np.zeros(k//2) # final part
    for i in range(k//2):
        erginit[i]=np.sum(data[:k//2+i+1])/k # since denominator ist still k, behaves like zero-padded data
        ergfinal[i]=np.sum(data[n-k//2-i-1:])/(k)
    erg=np.append(erg,ergfinal) # combine parts to one singal
    erg=np.insert(erg,0,erginit)
    return erg

def deriv(time,data): # Calculates the central point derivative of data over equidistantly sampled timepoints time

    h=time[5]-time[4]
    ddat=(data[2:]-data[0:-2])
    ddat=ddat/(2*h)
    ddat=np.insert(ddat,0,(data[1]-data[0])/h) # Take forward / backward derivative at start,end point
    ddat=np.append(ddat,(data[-1]-data[-2])/h)
    return ddat

def find_ccperiod(acctime,acc,window_len=1,cpr_thresh=28,freq=250,weightlen=1): # window_len...length of analysis window default=1s, cpr_thres...CPR threshold, freq=Frequency of acc
    acce=acc-np.mean(acc) # subtract constant gravitational acceleration
    
    
    dacctime=acctime[1:]-acctime[:-1] # Search for pauses in recording, fill them with zero entries
    dacctime=np.append(dacctime,1/freq) # add last element to get equal no of points
    gaps=np.argwhere(dacctime>0.1) # search for gaps in recording (indices)
    if len(gaps)!=0: # if there are any gaps
        zz=np.flip(np.concatenate(np.argwhere(dacctime>0.1))) # collect gaps
    else:
        zz=[]
    for i in zz:
        ngap=int(np.floor(freq*(+acctime[i+1]-acctime[i]-1/freq))) # length of gap
        acctime=np.insert(acctime,i+1,np.linspace(acctime[i]+1/freq,acctime[i+1],ngap,endpoint=False))
        acce=np.insert(acce,i+1,np.zeros(ngap)) 
        
    but=sg.butter(4,(0.2*2/freq,50*2/freq),btype='bandpass',output='sos')  # Bandpass filter signal
    acce=sg.sosfilt(but,acce)
    k=int(freq*window_len)
    softthres=10  # Soft shrinkage threshold
    avacc=av_mean(k,np.abs(acce)) # Average mean of abs(acc)
    davacc=deriv(acctime,avacc) # Derivative of average mean
    avdavacc=av_mean(k,davacc) # av_mean of derivative
    
    # Soft Thresholding to get rid of small extrema in derivative due to oscillations during cpr
    avdavacc=np.maximum(np.abs(avdavacc)-softthres,0)*np.sign(avdavacc) 
    
    n=len(acctime)
    peakmark=(avdavacc[2:]-avdavacc[1:-1])*(avdavacc[1:-1]-avdavacc[:-2]) # Determine peaks in averagre derivative (candidates)
    #end=np.arange(0,n-2)[(peakmark<=0) & (avdavacc[1:-1]<-thresh)] # possible stopping points
    pointcand=np.arange(0,n-2)[(peakmark<=0)] # possible starting poins
    #pointcand=np.sort(np.append(start,end))
    points={'Start':np.array([],int),'Stop':np.array([],int)} # construct container for start/end indices of cc periods
    
    flag=False # Start with search for starting point
    cand=0
    icand=0
    for i in pointcand:
        if not flag and avdavacc[i]<0: #while searching for start a stoplike value appears. save start value
            points['Start']=np.append(points['Start'],icand)
            flag = not flag
            cand=0
        elif flag and avdavacc[i]>0: #while searching for stop a startike value appears. save stop value
            points['Stop']=np.append(points['Stop'],icand)
            flag = not flag
            cand=0   
        if not flag: # Searching for start: Get maximum of derivative
            if avdavacc[i]>cand: # check whether derivative is larger here
                icand=i  # update values
                cand=avdavacc[icand]
        else: # Searching for end: Get minimum of derivative
            if avdavacc[i]<cand: # check whether derivative is smaller here
                icand=i   # update values
                cand=avdavacc[icand]
    if not flag: # add last point to start or endpoint (not in loop included)
        points['Start']=np.append(points['Start'],icand)
    else:
        points['Stop']=np.append(points['Stop'],icand)
    
    # Filter resulting CC periods after three conditions
    # 1st: in a pause the mean of absolute acceleration must be less than 35% of average of mean of cc periods before and after 
    badpoints=np.array([],int)
    for i in range(np.minimum(len(points['Start']),len(points['Stop']))-1): # Delete pauses, where average mean stays over 0.5 * mean(CPR Phase before and CPR phase after)
        pausethresh=0.35*0.5*(np.mean(avacc[points['Start'][i]:points['Stop'][i]])+np.mean(avacc[points['Start'][i+1]:points['Stop'][i+1]]))
        if np.min(avacc[points['Stop'][i]:points['Start'][i+1]])>pausethresh:     
            badpoints=np.append(badpoints,i) # Save indices

    points['Start']=np.delete(points['Start'],badpoints+1) # Delete indices
    points['Stop']=np.delete(points['Stop'],badpoints)

    # 2nd: # a CPR phase must last at least 1.6 seconds, delete shorter ones
    pauselen=1.6 # a CPR phase must last at least 1.6 seconds, delete shorter ones
    badpoints2=np.array([],int) 
    for i in range(np.minimum(len(points['Start']),len(points['Stop']))): # Delete CPR-Periods which are shorter then 2.5s
        if acctime[points['Stop'][i]]-acctime[points['Start'][i]]<pauselen:
            badpoints2=np.append(badpoints2,i)

    points['Start']=np.delete(points['Start'],badpoints2)
    points['Stop']=np.delete(points['Stop'],badpoints2)        
    
    
    # 3rd Acc_mean while cpr is not allowed to be below this threshold (unit presumably equivalent 2,8 inch /s^2, but units remain unclear)#   
    badpoints3=np.array([],int) 
    for i in range(np.minimum(len(points['Start']),len(points['Stop']))): # Delete CPR-Periods which are shorter then 2.5s
        if np.mean(avacc[points['Start'][i]:points['Stop'][i]])<cpr_thresh:
            badpoints3=np.append(badpoints3,i)

    points['Start']=np.delete(points['Start'],badpoints3)
    points['Stop']=np.delete(points['Stop'],badpoints3) 
    
    # Additionally correct index points by calculating a weighted mean
    nlen=int(0.5*weightlen*freq)
    for i in range(len(points['Start'])):
        elem=points['Start'][i]
        if elem>nlen and elem<n-nlen:
            points['Start'][i]=int(np.sum(avdavacc[elem-nlen:elem+nlen]*np.arange(elem-nlen,elem+nlen,1))/np.sum(avdavacc[elem-nlen:elem+nlen]))
           #print(elem,'    ', points['Start'][i])
    for i in range(len(points['Stop'])):
        elem=points['Stop'][i]
        if elem>nlen and elem<n-nlen:
            points['Stop'][i]=int(np.sum(avdavacc[elem-nlen:elem+nlen]*np.arange(elem-nlen,elem+nlen,1))/np.sum(avdavacc[elem-nlen:elem+nlen]))
    
    # convert indices to timestamps again
    alg_cc_period={}
    for key in points:
        alg_cc_period[key]=acctime[points[key]]
    return alg_cc_period

def plot_pauses(accel,alg_ccperiods,ann_ccperiods=[],xli=[]):
    acctime=np.array(accel['Time / s'])
    acc=np.array(accel['CPR Acceleration'])
    acc=acc-np.mean(acc)

    fig,ax=plt.subplots(1,figsize=(30,8))
    ax.plot(acctime,acc,color='indigo',alpha=0.3,label='Acceleration Signal without gravity')
    ax.grid(True)
    ax.vlines(alg_ccperiods['Start'],-1000,1000,color='orange',alpha=0.5,ls='--',label='Alg CC-period Start',lw=4)
    ax.vlines(alg_ccperiods['Stop'],-1000,1000,color='red',alpha=0.5,ls='--',label='Alg CC-period Stop',lw=4)
    if len(ann_ccperiods)!=0:
        ax.vlines(ann_ccperiods['CC-period-start'],-1000,1000,color='green',alpha=0.5,ls=':',label='Ann CC-period Start',lw=3)
        ax.vlines(ann_ccperiods['CC-period-stop'],-1000,1000,color='blue',alpha=0.5,ls=':',label='Ann CC-period Stop',lw=3)
    ax.legend()
    ax.set_ylim(-500,500)
    if len(xli)==2:
        ax.set_xlim(xli)    
    else:
        ax.set_xlim(acctime[0],acctime[-1])
    fig.show()
    