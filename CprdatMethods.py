#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:10:15 2021

@author: kernwo
"""

import numpy as np
import scipy.signal as sg
import scipy.fft as fft
import matplotlib.pyplot as plt
import joblib 
from matplotlib.lines import Line2D
import scipy.stats as stats
import antropy as ant




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


def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def mean_prob(tim,prob,le=15,sigma=5):
    statim=np.min(tim)
    stotim=np.max(tim)
    timrange=np.arange(statim,stotim,2)
    prob_mean=np.zeros(len(timrange))
    prob_std=np.zeros(len(timrange))
    for i,elem in enumerate(prob_mean):
        dat=prob[(timrange[i]-tim<le)&((timrange[i]-tim)>=-le)]
        timdat=timrange[i]-tim[(timrange[i]-tim<le)&((timrange[i]-tim)>=-le)]

        nn=len(dat)
        if nn>=2:
            weights=normal_dist(timdat,0,sigma)
            weights=weights/np.sum(weights)
            prob_mean[i]=np.sum(weights*dat)
            prob_std[i]=np.sqrt(np.sum(weights*np.square(dat))-np.square(np.sum(weights*dat)))
        else:
            prob_mean[i]=np.nan
            prob_std[i]=np.nan
    return timrange,prob_mean,prob_std

# ---------------------------FOR CC-Periods------------------------------------------

def find_ccperiod(acctime,acc,window_len=1,cpr_thresh=28,freq=250,weightlen=1,return_index_flag=False): # window_len...length of analysis window default=1s, cpr_thres...CPR threshold, freq=Frequency of acc
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
    if return_index_flag:
        return points
    else:
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
    
    
# ------------------------------ FOR MECHANIC ACTIVITY PLOTS ------------------------------------    
    
    
def max_nontriv_autocorr(acctime,acc): # Computes the nontrivial maximaum of the autocorrelation of signal
    sniplen=len(acc)
    z=np.correlate(acc,acc,mode='same')/(np.sum(np.square(acc)))
    a=z[2:]-z[1:-1]
    b=z[1:-1]-z[:-2]
    j=np.argwhere((b>0)&(a<0)).flatten()+1
    j=j[np.abs(j-sniplen//2)>15]
    ze_max=0
    j_max=0
    for jj in j:
        if z[jj]>ze_max:
            ze_max=z[jj]
            j_max=jj
    
    return acctime[j_max]-2,ze_max


def qrs_detector(ekg):
    ekg=np.array(ekg)
    but2=sg.butter(4,(0.5/125,30/125),output='sos',btype='bandpass')
    ek_filt=sg.sosfilt(but2,ekg)

    d_filt=av_mean(25,np.square(ek_filt[1:]-ek_filt[:-1])) #0.1 s average mean
    d_filt=np.append(d_filt,0)
    d_filt=d_filt/np.max(d_filt)
    # Arhy Correlation via search for maxs
    d_filt2=av_mean(25,np.square(ek_filt[1:]-ek_filt[:-1])) # Search for QRS-komplexes ( idea after irusta above)
    d_filt2=np.append(d_filt2,0)
    d_filt2=d_filt2/np.max(d_filt2)

    max2=np.argwhere((d_filt2[2:]-d_filt2[1:-1]<0) & (d_filt2[1:-1]-d_filt2[:-2]>0) & \
                     (d_filt2[1:-1]>0.33)).flatten()+1 # Search for maxima exceeding 0.33

    max3=np.argwhere((ek_filt[2:]-ek_filt[1:-1])*(ek_filt[1:-1]-ek_filt[:-2])<0).flatten()+1
    # Compute true maxs by taking the largest local maximum within 50*0.004=0.200 seconds
    true_max=np.array([max3[0]])
    for elem in max2:
        cand=max3[np.abs(max3-elem)<75]
        if len(cand)==0:
            cand=np.array([max3[np.argmin(np.abs(max3-elem))]])
        ek_cand=np.abs(ek_filt[cand])

        true_cand=cand[np.argmax(ek_cand)]
        if (true_cand-true_max[-1])<75:
            if np.abs(ek_filt[true_cand]) >np.abs(ek_filt[true_max[-1]]):
                true_max[-1]=true_cand
        else:
            true_max=np.append(true_max,true_cand)
    
    n=len(ekg)
    le=60
    true_max=true_max.astype(int)
    true_max=true_max[(true_max>le)&(true_max<n-le)] 
    qq=np.array([],dtype=int)
    ss=np.array([],dtype=int)
    for r in true_max:
        ii=np.where(max3==r)[0][0]
        q_cand=max3[ii-4:ii]
        q_cand=q_cand[r-q_cand<100]
        if len(q_cand)==0:
            q=0
        else:
            if ek_filt[r]>0:
                q=q_cand[np.argmin(ek_filt[q_cand])]
            if ek_filt[r]<0:
                q=q_cand[np.argmax(ek_filt[q_cand])]
        s_cand=max3[ii+1:ii+5]
        s_cand=s_cand[s_cand-r<100]
        if len(s_cand)==0:
            s=0
        else:
            if ek_filt[r]>0:
                s=s_cand[np.argmin(ek_filt[s_cand])]
            if ek_filt[r]<0:
                s=s_cand[np.argmax(ek_filt[s_cand])]
        qq=np.append(qq,q)
        ss=np.append(ss,s)
    return true_max,qq,ss

def acc_feature(acc,acc_ensemble): # Compute all ACC Features from Ashouri 2017
    if acc_ensemble.shape[0]==120:
        accm=acc_ensemble
    else:
        accm=np.mean(acc_ensemble,axis=0)
    acc_rms=np.sqrt(np.sum(np.square(accm))/120)
    acc_kurt=stats.kurtosis(accm)
    acc_skew=stats.skew(accm)
    acc_median=np.median(accm)
    acc_pp=np.max(accm)-np.min(accm)
    acc_pprms_ratio=acc_pp/acc_rms
    
    freq,psd=sg.welch(accm,fs=250,nperseg=120)
    psd=psd/np.sqrt(np.sum(np.square(psd)))
    cond=(freq>0.8) & (freq<30)
    psd=psd[cond]
    freq=freq[cond]
    
    power_bands=[]
    for freq_thresh in [(0,3),(3,6),(6,9),(9,12),(12,15),(15,18)]:
        fl,fu=freq_thresh
        cond=(freq>=fl) & (freq<fu)
        psd_thres=psd[cond]
        power_bands.append(np.sqrt(np.sum(np.square(psd_thres))))

    freq,psd=sg.welch(accm,fs=250,nperseg=120)
    cond=(freq>0.8) & (freq<30)
    psd=psd[cond]
    freq=freq[cond]
    psd_mean=np.mean(psd)
    psd_std=np.std(psd)
    psd_kurt=stats.kurtosis(psd)
    psd_skew=stats.skew(psd)
    psd_max=np.max(psd)
    psd_maxfreq=freq[np.argmax(psd)]
    acc_feature_list=[acc_rms,acc_kurt,acc_skew,acc_median,acc_pp,acc_pprms_ratio,power_bands,psd_mean,psd_std,psd_kurt,psd_skew,psd_max,psd_maxfreq]
    return acc_feature_list

def ek_feature(ekg): # Compute a variety of ecg-features (mostly Elola 2020)
    n=len(ekg)
    but2=sg.butter(4,(0.5/125,30/125),output='sos',btype='bandpass')
    ek_filt=sg.sosfilt(but2,ekg)

    # Arhy Correlation via search for maxs
    d_filt2=av_mean(25,np.square(ek_filt[1:]-ek_filt[:-1])) # Search for QRS-komplexes ( idea after irusta above)
    d_filt2=np.append(d_filt2,0)
    d_filt2=d_filt2/np.max(d_filt2)

    true_max,qq,ss=qrs_detector(ekg)

    pp_amp=np.array([])
    qs_width=np.array([])

    for q,s in zip(list(qq),list(ss)):
        ek=ekg[q:s+1]
        if len(ek)>0:
            pp_amp=np.append(pp_amp,np.max(ek)-np.min(ek))
            qs_width=np.append(qs_width,(s-q)*0.004)
        else:
            pp_amp=np.append(pp_amp,0)
            qs_width=np.append(qs_width,0)

    if len(true_max)>1:
        pp_mean=np.mean(pp_amp)
        pp_std=np.std(pp_amp)
        qrs_width_mean=np.mean(qs_width)
        qrs_width_std=np.std(qs_width)
        qrs_pp_width_ratio=np.median(pp_amp/(qs_width+1e-16))
    else:
        pp_mean=0
        pp_std=0
        qrs_width_mean=0
        qrs_width_std=0
        qrs_pp_width_ratio=0


    if len(true_max)>=2:
        rr_list=(true_max[1:]-true_max[:-1])*0.004
        rr_mean=np.mean(rr_list)
        rr_std=np.std(rr_list)
    else:
        rr_mean=4
        rr_std=0

    ek_filt2=(ek_filt-np.min(ek_filt))/(np.max(ek_filt)-np.min(ek_filt))
    ecg_norm_median=np.median(ek_filt2)
    ecg_norm_var=np.var(ek_filt2)

    d_filt=ek_filt[1:]-ek_filt[:-1] #0.1 s average mean
    d_filt=np.append(d_filt,0)
    #d_filt=d_filt/np.max(d_filt)

    slope_mean=np.mean(np.abs(d_filt))
    slope_std=np.std(np.abs(d_filt))
    slope_kurt=stats.kurtosis(d_filt2)

    s_ecg=np.abs(fft.fft(ek_filt*sg.windows.tukey(1000,alpha=0.0),norm='ortho')[:n//2])
    freq=fft.fftfreq(n,0.004)[:n//2]
    cond=(freq>2) & (freq<30)
    AMSA=np.sum(freq[cond]*s_ecg[cond])

    cond=(freq>17.5) & (freq<30)
    ecg_fib_pow=np.sum(np.square(s_ecg[cond]))#*250/2000

    hjmob,hjcomp=ant.hjorth_params(ek_filt)
    ecg_skew=stats.skew(ek_filt)
    ek_class2=[ rr_mean, rr_std,pp_mean,pp_std,qrs_width_mean,qrs_width_std,qrs_pp_width_ratio,ecg_norm_median,ecg_norm_var, 
        slope_mean,slope_std,slope_kurt, AMSA, ecg_fib_pow,hjmob,hjcomp,ecg_skew]
    return ek_class2

    
def ekg_acc_corr1(acctime,acc,ekg,nperse=1000,window=''): # Compute Features
    n=len(acc)
    w=np.ones(n)
    if window=='hamming':
        w=sg.windows.hamming(n)
    elif window=='hann':
        w=sg.windows.hann(n)

    

    acc_amp=np.sqrt(np.sum(np.square(acc))) # Mean of absolute values (v_1)
    ekg_amp=np.sqrt(np.sum(np.square(ekg))) # Mean of absolute values (v_2)
    
    # Compute Spectral overlap of Frequencies between ftr2 and ftr
    sp_cor=[]
    ftr=20.01 ## Frequency Threshold    
    ftr2=.75
  
    Sdd=np.abs(fft.fft(acc*w,norm='ortho')[:n//2]) # FFT of Acc
    See=np.abs(fft.fft(ekg*w,norm='ortho')[:n//2]) # FFT of ECG
    freq0=fft.fftfreq(nperse,0.004)[:n//2] # Freqs for FFT    

    Sdd=Sdd[(freq0<ftr)&(freq0>ftr2)] # Include only frequencies below threshold
    See=See[(freq0<ftr)&(freq0>ftr2)]
    
    Sdd=Sdd/np.sqrt(np.sum(np.square(np.abs(Sdd)))) # Normalize Spectra
    See=See/np.sqrt(np.sum(np.square(np.abs(See))))
    
    
    acc_v=np.abs(Sdd)-np.mean(np.abs(Sdd)) # Subtract mean from spectra
    ekg_v=np.abs(See)-np.mean(np.abs(See))

    acc_st2=np.sum(np.square(acc_v)) #compute absolute of spectra
    ekg_st2=np.sum(np.square(ekg_v))
    sp_cor=np.sum(acc_v*ekg_v)/np.sqrt(acc_st2*ekg_st2) # Compute spectral overlap ( Covariance of Spectra) v_3

    
    # Use Autocorrelation methods
    z=np.correlate(acc,acc,mode='same')/(np.sum(np.square(acc))) # Acc Autocorrelation
    ze=np.correlate(ekg,ekg,mode='same')/(np.sum(np.square(ekg))) # ECG Autocorrelation
    
    a=ze[2:]-ze[1:-1] # First Difference forward
    b=ze[1:-1]-ze[:-2] # First Difference backward
    j=np.argwhere((b>0)&(a<0)).flatten() # Search maxima of ecg autocorr
    j=j[(np.abs(j-n//2)>75)&(j>12) & (j<n-12)] # Ignore maxima in the middle (trivial and at the boundaries)
    
    # Search for largest maximum of ECG autocorr
    ze_max=-1
    j_max=0
    for jj in j:
        if ze[jj]>ze_max:
            ze_max=ze[jj] #v_4
            j_max=jj
    if len(j)==0:
        j_max=250
        
    z_max=z[j_max] # Acc Autocorr @ largest ECG autocorr v_6
    d2zz=(z[j_max+1-2:j_max+1+3]+z[j_max-1-2:j_max-1+3]-2*z[j_max-2:j_max+3])*250**2 # second derivative at this point
    d2z=np.nanmean(d2zz)   # v_7
    #d2z=(z[j_max+1]+z[j_max-1]-2*z[j_max])*250**2
    za_max=max_nontriv_autocorr(acctime,acc)[1] # Maximal autocorr of acceleration only v_5
    
    
    
    
    # Features nach Irusta 2014 "A Reliable Method for Rhythm Analysis during Cardiopulmonary Resuscitation"
    but=sg.butter(5,2.5/125,output='sos',btype='highpass')
    ek_lea=sg.sosfilt(but,ekg)

    P_lea=np.sum(np.square(ek_lea))
    Lk=np.array([])
    for k in range(8):
        Lk=np.append(Lk,np.sum(np.sqrt(np.square(ek_lea[k*125:(k+1)*125])+0.004**2)))
    Lmin=np.min(Lk)

    but2=sg.butter(4,(0.5/125,30/125),output='sos',btype='bandpass')
    ek_filt=sg.sosfilt(but2,ekg)

    d_filt=av_mean(25,np.square(ek_filt[1:]-ek_filt[:-1])) #0.1 s average mean
    d_filt=np.append(d_filt,0)
    d_filt=d_filt/np.max(d_filt)

    bS=np.percentile(d_filt,10)
    maxs=np.argwhere((d_filt[2:]-d_filt[1:-1]<0) & (d_filt[1:-1]-d_filt[:-2]>0) & (d_filt[1:-1]>0.2)).flatten()+1
    nP=len(maxs)
    
    ffe0=np.abs(fft.fft(ekg*sg.windows.hamming(n),norm='ortho')[:n//2])
    ffe0=ffe0/np.sqrt(np.sum(np.square(np.abs(ffe0))))
    
    P_fib=np.sum(np.square(ffe0[(freq0>=2.5)&(freq0<=7.5)]))
    P_h=np.sum(np.square(ffe0[(freq0>=12)]))
    
    ek_classifiers=[P_lea,Lmin,bS,nP,P_fib,P_h]
    
    
    # Arhy Correlation via search for maxs
    d_filt2=av_mean(25,np.square(ek_filt[1:]-ek_filt[:-1])) # Search for QRS-komplexes ( idea after irusta above)
    d_filt2=np.append(d_filt2,0)
    d_filt2=d_filt2/np.max(d_filt2)

    max2=np.argwhere((d_filt2[2:]-d_filt2[1:-1]<0) & (d_filt2[1:-1]-d_filt2[:-2]>0) & \
                     (d_filt2[1:-1]>0.33)).flatten()+1 # Search for maxima exceeding 0.33

    # Compute true maxs by taking the largest local maximum within 50*0.004=0.200 seconds
    true_max=np.array([max2[0]])
    j=d_filt2[max2[0]]
    for elem in max2:
        if (elem-true_max[-1])<50:
            if d_filt2[elem] >d_filt2[true_max[-1]]:
                true_max[-1]=elem
        else:
            true_max=np.append(true_max,elem)

    le=60
    true_max=true_max.astype(int)
    true_max=true_max[(true_max>le)&(true_max<n-le)] # Take only maxima, which allow cutting a window around them 
    # not to close to the border

    i=1
    # Compute Correlation between all windows
    corr_arhy=np.array([])
    corr_arhy_ekg=np.array([])

    for beat in true_max:
        for beat2 in true_max[i:]:
            ac1=acc[beat-le:beat+le]
            ac2=acc[beat2-le:beat2+le]
            corr_arhy=np.append(corr_arhy,np.sum(ac1*ac2)/(np.sqrt(np.sum(np.square(ac1))*np.sum(np.square(ac2)))))
            
            ek1=ekg[beat-le:beat+le]
            ek2=ekg[beat2-le:beat2+le]
            corr_arhy_ekg=np.append(corr_arhy_ekg,np.sum(ek1*ek2)/(np.sqrt(np.sum(np.square(ek1))*np.sum(np.square(ek2)))))
            
        i+=1
        # take 3rd quartile of this quantity
    if len(corr_arhy)>0:
        arhy_cor=np.percentile(corr_arhy,75)
    else:
        arhy_cor=0
    if len(corr_arhy_ekg)>0:
        arhy_cor_ekg=np.percentile(corr_arhy_ekg,75)
        arhy_cor_quot=arhy_cor/arhy_cor_ekg
    else:
        arhy_cor_ekg=0
        arhy_cor_quot=0
    
    
    # make ensemble average
    
    if len(true_max)>0:
        for i,beat in enumerate(true_max):
            ac1=acc[beat-le:beat+le]
            if i==0:
                acc_ensemble=np.array(ac1)
            else:
                acc_ensemble=np.vstack([acc_ensemble,ac1])
    else:
        acc_ensemble=np.random.randn(5,120)
    spec_ent_acc=ant.spectral_entropy(acc,sf=250,normalize=True,method='welch')
        
    ac_f=acc_feature(acc,acc_ensemble)
    ek_f=ek_feature(ekg)
    
    return sp_cor ,acc_amp,ekg_amp,z_max,ze_max,za_max,d2z,arhy_cor,arhy_cor_ekg,arhy_cor_quot, spec_ent_acc,ek_classifiers,ac_f,ek_f

def collect_data(accel,ecg,co,shocks,physio):
    roscs=np.array(physio['ROSC / Termination'])
    rearrest=np.array(physio['Arrest'])

    acc=np.array(accel['CPR Acceleration']) # convert acceleration data into numpy arrays
    acctime=np.array(accel['Time / s'])
    ekgtime=np.array(ecg['Time / s'])
    ekg=np.array(ecg['Pads'])
    co2=np.array(co['CO2 mmHg, Waveform'])
    co2time=np.array(co['Time / s'])

    shock_time=np.array(shocks['timestamp'])

    signals=[acctime,acc,ekgtime,ekg,co2time,co2]
    annot=[roscs,rearrest]
    data=[signals,annot,shock_time]
    return data


def construct_data_for_snippets(data):
    sniplen=1000
    signals,annot,shock_time=data
    roscs,rearrest=annot
    acctime,acc,ekgtime,ekg,co2time,co2=signals

    starttime=np.max([acctime[0],ekgtime[0],co2time[0]])
    stoptime=np.min([acctime[-1],ekgtime[-1],co2time[-1]])

    acc=acc[(acctime >starttime) & (acctime <stoptime)]
    acctime=acctime[(acctime >starttime) & (acctime <stoptime)]
    ekg=ekg[(ekgtime >starttime) & (ekgtime <stoptime)]
    ekgtime=ekgtime[(ekgtime >starttime) & (ekgtime <stoptime)]
    co2=co2[(co2time >starttime) & (co2time <stoptime)]
    co2time=co2time[(co2time >starttime) & (co2time <stoptime)]  

    acc=acc-np.mean(acc)

    points=find_ccperiod(acctime,acc,return_index_flag=True)
    points['Start']=np.append(points['Start'],len(acctime)-1) # classify last interval as no CPR (since succeeding marker is CPR Start)
    points['Stop']=np.insert(points['Stop'],0,0) # classify first interval as no CPR (since marker before is no CPR Stop)
    le=np.min([len(points['Start']),len(points['Stop'])])
    points['Start']=points['Start'][:le]
    points['Stop']=points['Stop'][:le]

    i=0
    sniplen=1000
    features=['Acc Amp', 'Pure Acc Corr', 'Spectral Entropy ACC', 'Mean ACC Power', 'Mean ACC Kurtosis', 'Mean ACC Skewness', 'Mean ACC Median', 'Mean ACC PeaktoPeak',
              'Mean ACC PP-RMS-Ratio', 'PSD_03_Band', 'PSD_36_Band', 'PSD_69_Band', 'PSD_912_Band', 'PSD_1215_Band', 'PSD_1518_Band', 'PSD_Mean', 'PSD_Std', 'PSD_Kurt',
              'PSD_Skew', 'PSD_Max1', 'PSD_Maxfreq1', 
              'Spectral OV', 'Acc(ECG) Corr', 'd2 AccCorr', 'Arhy Corr', 'Arhy Corr Quotient',
              'ECG Amp', 'Pure ECG Corr', 'Power LEA', 'Min ECG Length', 'bS', 'number Peaks', 'Fibrillation Power', 'High Freq Power', 'Arhy Corr EKG', 'RR_Mean', 
              'RR_Std', 'QRS_PP_Mean',  'QRS_PP_Std', 'QRS_width_mean', 'QRS_width_std', 'QRS_heigth_width_ratio', 'ECG_slope_mean', 'ECG_slop_Std', 
              'ECG_slop_Kurtosis', 'AMSA', 'ECG_Fib_power',  'ECG_norm_median', 'ECG_norm_var']#,'Hjorth_mob','Hjorth_comp','ECG_skew']# LIST of all Features
    background_infos=['ACC', 'ECG', 'CO', 'File', 'Start Time', 'Start Distance', 'End Distance',\
                      'Max ACC','Max ECG','Max CO2','ACC ratio','ECG ratio','Shock before','Shock after','Time_since_CC','Rhythm']
    snippets={'Snippet':{},'Type':[],'Analysis':{}} # Contruct key
    snippets['Analysis']=dict([(feature,[]) for feature in features])
    snippets['Snippet']=dict([(typ,[]) for typ in background_infos])


    for i in range(len(points['Start'])): # Iteration over all start points of CPR Episodes
        istart=np.max([points['Start'][i]-200,0]) # End Pause (Start CPR) points['Start'][i]-125
        istop=np.min([points['Stop'][i]+200,len(acctime)-1]) # Begin Pause (Stop CPR), points['Stop'][i]+125
        #print('Interval',istop,istart)
        if istart-istop>sniplen: # #If Pause is long enouth to contain a snippet

            n_snip=(istart-istop)//(sniplen//2)-2 # Number of Snippets in pause
            r=(istart-istop) % sniplen # Remainder
            for j in range(n_snip): # ' iterate over all Snippets'
                snip_i=istop+r//2+j*sniplen//2
                snip_f=istop+r//2+(j+2)*sniplen//2
                t_since_cc=(snip_i-istop)/250
                # Check whether actual snippet is a ROSC, Arrest of End Snippet (End.. check outcome of CPR attempt in registry)

                typo=''            

                if snip_f<rearrest[0]:
                    typo='SC'
                else:
                    for ire,ros in enumerate(roscs):  
                        rea=rearrest[ire]
                        if acctime[snip_i]>rea and acctime[snip_f]<ros:
                            # If begin of snippet  and end of snippet after last arrest marker and before next rosc marker and 
                            typo='AR'
                            start_dif=acctime[snip_i]-rea
                            end_dif=ros-acctime[snip_i]
                        elif ire<len(rearrest)-1 and acctime[snip_i]>ros and acctime[snip_f]<rearrest[ire+1]:
                            # If snippet totally between rosc and arrest marker
                            typo='SC'
                            start_dif=acctime[snip_i]-ros
                            end_dif=rearrest[ire+1]-acctime[snip_i]
                        elif ire==len(rearrest)-1 and acctime[snip_i]>ros:
                            #if type(case_GRR['ZUEBG'].iloc[0])==float:
                            typo='End'#'ROSC-End'
                            start_dif=acctime[snip_i]-ros
                            end_dif=np.nan
                            ##elif type(case_GRR['ZTOD'].iloc[0])==float:
                            #start_dif=acctime[snip_i]-ros
                            #end_dif=np.nan
                            #typo='Arrest-End'
                            
                if (typo=='SC') or (typo=='End'):
                    cos=co2[(co2time>=acctime[snip_i]) & (co2time<acctime[snip_f])]
                    if len(cos)==0:
                        typo=''
                #    elif np.max(cos)<20:
                #        typo=''
                if (typo=='AR') or (typo=='AR-End'):
                    cos=co2[(co2time>=acctime[snip_i]) & (co2time<acctime[snip_f])]
                    if len(cos)==0:
                        typo=''
                #    elif np.max(cos)>30:
                #        typo=''
                acce=acc[snip_i:snip_f]
                ekge=ekg[snip_i:snip_f]
                acce-=np.mean(acce)
                ekge-=np.mean(ekge)
                if (typo!='') :
                    
                    
                    snippets['Snippet']['ACC'].append(acc[snip_i:snip_f])
                    snippets['Snippet']['ECG'].append(ekg[snip_i:snip_f])
                    snippets['Snippet']['CO'].append(cos)#,'ECG':ekg[snip_i:snip_f],'CO':cos})
                    #snippets['Snippet']['File'].append(caseno)
                    snippets['Snippet']['Start Time'].append(acctime[snip_i])
                    snippets['Snippet']['Start Distance'].append(start_dif)
                    snippets['Snippet']['End Distance'].append(end_dif)
                    snippets['Snippet']['Time_since_CC'].append(t_since_cc)


                    snippets['Snippet']['Max ACC'].append(np.max(np.abs(acce)))
                    snippets['Snippet']['Max ECG'].append(np.max(np.abs(ekge)))
                    snippets['Snippet']['Max CO2'].append(np.max(np.abs(cos)))
                    snippets['Snippet']['ACC ratio'].append(np.max(np.abs(acce))/np.mean(np.abs(acce)))
                    snippets['Snippet']['ECG ratio'].append(np.max(np.abs(ekge))/np.mean(np.abs(ekge)))
                    snippets['Type'].append(typo)

                    (sp_cor ,acc_amp,ekg_amp,z_max,ze_max,za_max,d2z,arhy_cor,arhy_cor_ekg,arhy_cor_quot, 
                     spec_ent_acc,ekg_feat,ac_f,ek_f)=ekg_acc_corr1(np.arange(0,sniplen*0.004,0.004),
                                                                    acce,ekge,nperse=sniplen)
                    
                    snippets['Analysis']['Spectral OV'].append(sp_cor)
                    snippets['Analysis']['Acc Amp'].append(acc_amp)
                    snippets['Analysis']['ECG Amp'].append(ekg_amp)
                    snippets['Analysis']['Pure ECG Corr'].append(ze_max)
                    snippets['Analysis']['Acc(ECG) Corr'].append(z_max)
                    snippets['Analysis']['d2 AccCorr'].append(d2z)
                    snippets['Analysis']['Pure Acc Corr'].append(za_max)
                    snippets['Analysis']['Arhy Corr'].append(arhy_cor)
                    snippets['Analysis']['Arhy Corr EKG'].append(arhy_cor_ekg)
                    snippets['Analysis']['Arhy Corr Quotient'].append(arhy_cor_quot)
                    snippets['Analysis']['Spectral Entropy ACC'].append(spec_ent_acc)
    
                    (acc_rms,acc_kurt,acc_skew,acc_median,acc_pp,acc_pprms_ratio,power_bands,
                     psd_mean,psd_std,psd_kurt,psd_skew,psd_max,psd_maxfreq)=ac_f
                    snippets['Analysis']['Mean ACC Power'].append(acc_rms)
                    snippets['Analysis']['Mean ACC Kurtosis'].append(acc_kurt)
                    snippets['Analysis']['Mean ACC Skewness'].append(acc_skew)
                    snippets['Analysis']['Mean ACC Median'].append(acc_median)
                    snippets['Analysis']['Mean ACC PeaktoPeak'].append(acc_pp)
                    snippets['Analysis']['Mean ACC PP-RMS-Ratio'].append(acc_pprms_ratio)
    
                    psd0,psd3,psd6,psd9,psd12,psd15=power_bands 
                    snippets['Analysis']['PSD_03_Band'].append(psd0)
                    snippets['Analysis']['PSD_36_Band'].append(psd3)
                    snippets['Analysis']['PSD_69_Band'].append(psd6)
                    snippets['Analysis']['PSD_912_Band'].append(psd9)
                    snippets['Analysis']['PSD_1215_Band'].append(psd12)
                    snippets['Analysis']['PSD_1518_Band'].append(psd15)
    
                    snippets['Analysis']['PSD_Mean'].append(psd_mean)
                    snippets['Analysis']['PSD_Std'].append(psd_std)
                    snippets['Analysis']['PSD_Kurt'].append(psd_kurt)
                    snippets['Analysis']['PSD_Skew'].append(psd_skew)
                    snippets['Analysis']['PSD_Max1'].append(psd_max)
                    snippets['Analysis']['PSD_Maxfreq1'].append(psd_maxfreq)
    
                    [P_lea,Lmin,bS,nP,P_fib,P_h]=ekg_feat
                    snippets['Analysis']['Power LEA'].append(P_lea)
                    snippets['Analysis']['Min ECG Length'].append(Lmin)
                    snippets['Analysis']['bS'].append(bS)
                    snippets['Analysis']['number Peaks'].append(nP)
                    snippets['Analysis']['Fibrillation Power'].append(P_fib)
                    snippets['Analysis']['High Freq Power'].append(P_h)      
    
                    ekf_keys=['RR_Mean', 'RR_Std', 'QRS_PP_Mean', 'QRS_PP_Std', 'QRS_width_mean', 'QRS_width_std', 'QRS_heigth_width_ratio', 
                              'ECG_norm_median', 'ECG_norm_var','ECG_slope_mean', 'ECG_slop_Std', 'ECG_slop_Kurtosis', 'AMSA', 'ECG_Fib_power' ]
                    for key,value in zip(ekf_keys,ek_f):
                        snippets['Analysis'][key].append(value)      
 
    return snippets



    
def data_filter(X,X_background,y,accthresh=20,ecgthresh=2.5,accr=25,ecgr=35,arresttimemin=None,rosctimemin=None,wo_end=False,ros_co=None,are_co=None,shock_before=None,shock_after=None,
               background_keys=['Start Time', 'Start Distance', 'End Distance', 'Max ACC', 'Max ECG', 'Max CO2', 'ACC ratio', 'ECG ratio','Shock before','Shock after','Rhythm']):
    # Filters given features X, background infos X_background and labels y after conditions given in background infos 
    cond=np.ones(X.shape[0],dtype=bool)
    #print(len(cond[cond==1]))
    i=background_keys.index('Max ACC')
    cond=(cond) & (X_background.T[i]<accthresh) # max(abs(acc)) during a Snippet must be smaller than accthresh # Remove Noisy data
    #print(len(cond[cond==1]))

    i=background_keys.index('Max ECG')
    cond=(cond) & (X_background.T[i]<ecgthresh) # max(abs(ecg)) during a Snippet must be smaller than ecgthresh # Remove Noisy data
    #print(len(cond[cond==1]))

    i=background_keys.index('ACC ratio')
    cond=(cond) & (X_background.T[i]<accr) # max(abs(acc))/median(abs(acc)) during a Snippet must be smaller than accr. # Remove data where amplitudes are sometimes way bigger than otherwhere (sudden peaks etc)
    #print(len(cond[cond==1]))

    i=background_keys.index('ECG ratio')  # max(abs(ecg))/median(abs(ecg)) during a Snippet must be smaller than ecgr. # Remove data where amplitudes are sometimes way bigger than otherwhere (sudden peaks etc)
    cond=(cond) & (X_background.T[i]<ecgr)
    #print(len(cond[cond==1]))
    
    if arresttimemin!=None:     # Remove snippets which are less than arresttimemin seconds distance from a arrest marker. 
        i=background_keys.index('End Distance')
        j=background_keys.index('Start Distance')
        X_background.T[i]
        cond=(cond) & (((y.astype(int)==1) & (X_background.T[i]>arresttimemin)) | ((y.astype(int)==-1) & (X_background.T[j]>arresttimemin)))
    #print(len(cond[cond==1]))
    if rosctimemin!=None:     # Remove snippets which are less than arresttimemin seconds distance from a arrest marker. 
        i=background_keys.index('End Distance')
        j=background_keys.index('Start Distance')
        X_background.T[i]
        cond=(cond) & (((y.astype(int)==1) & (X_background.T[j]>rosctimemin)) | ((y.astype(int)==-1) & (X_background.T[i]>rosctimemin)))
    #print(len(cond[cond==1]))    
    if (shock_before!=None) & (shock_after!=None): 
        i=background_keys.index('Shock before')
        i2=background_keys.index('Shock after')
        cond=(cond) & ((X_background.T[i]<shock_before) | (X_background.T[i2]<shock_after)) # Include only snippets which are less than shock_before seconds away from the next Shock
    #print(len(cond[cond==1]))
    
    if wo_end==True:
        cond=cond & ((y==-1) | (y==1))
    #print(len(cond[cond==1]))

    if ros_co!=None or are_co!=None:
        i=background_keys.index('Max CO2')
        cond=cond & (((y.astype(int)==1) & (X_background.T[i]>ros_co)) | ((y.astype(int)==-1) & (X_background.T[i]<are_co)))
    #print(len(cond[cond==1]))
    
    X=(X[cond])
    X_background=(X_background[cond])
    y=y[cond]
    
    
    return X,X_background,y

def predict_circulation(erg):
    acc_features=['Acc Amp','Pure Acc Corr','Spectral Entropy ACC','Mean ACC Power','Mean ACC Kurtosis','Mean ACC Skewness',
                  'Mean ACC Median','Mean ACC PeaktoPeak','Mean ACC PP-RMS-Ratio','PSD_03_Band','PSD_36_Band','PSD_69_Band',
                  'PSD_912_Band','PSD_1215_Band','PSD_1518_Band','PSD_Mean','PSD_Std','PSD_Kurt','PSD_Skew','PSD_Max1','PSD_Maxfreq1',
                  'Spectral OV','Acc(ECG) Corr','d2 AccCorr','Arhy Corr','Arhy Corr Quotient']
    ecg_features=['ECG Amp', 'Pure ECG Corr','Power LEA', 'Min ECG Length', 'bS', 'number Peaks', 'Fibrillation Power', 'High Freq Power', 'Arhy Corr EKG', 
                  'RR_Mean', 'RR_Std', 'QRS_PP_Mean', 
                  'QRS_PP_Std', 'QRS_width_mean', 'QRS_width_std', 'QRS_heigth_width_ratio', 'ECG_slope_mean', 'ECG_slop_Std',
                  'ECG_slop_Kurtosis', 'AMSA', 'ECG_Fib_power', 
                  'ECG_norm_median', 'ECG_norm_var']#,'Hjorth_mob','Hjorth_comp','ECG_skew'] #'Spectral Entropy ECG',
    all_features=acc_features+ecg_features
    train_keys=all_features
    
    background_keys=['Start Time', 'Start Distance', 'End Distance', 'Max ACC', 'Max ECG', 'Max CO2', 'ACC ratio', 'ECG ratio']
    
    # Construct data arrays X and X_background for all features and all necessary background information, as well as label vector y
    N_total=len(erg['Type'])
    n_f=len(train_keys)
    X=np.empty((N_total,n_f))
    y=np.empty(N_total)
    X_background=np.empty((N_total,len(background_keys)+1))

    for i in range(N_total):
        k=0
        typ=erg['Type'][i]

        for key in train_keys:
            X[i,k]=erg['Analysis'][key][i]
            k+=1
        X_background[i,-1]=i
        if (typ=='SC'):# or (typ=='SC-End'):
            y[i]=1
        elif (typ=='AR'):# or (typ=='AR-End'):
            y[i]=-1
        elif typ=='End': # Label for 'End' part unkown
            y[i]=0
        k=0
        for key in background_keys:
            X_background[i,k]=erg['Snippet'][key][i]
            k+=1
    y=y.astype(int)
    
    # Prefilter data
    Xn,Xn_background,yn=data_filter(X,X_background,y,background_keys=background_keys)
    # Load model and scaler
    svc=joblib.load('ml-models/circ-classification_model.joblib')
    scaler=joblib.load('ml-models/circ-classification_scaler.joblib')
    # Apply scaler
    Xn=scaler.transform(Xn)
    X=scaler.transform(X)
    
    # Predict results
    y_pred=svc.predict(X) # Xnn
    dec=svc.decision_function(X)
    y_proba=svc.predict_proba(X)
    
    # Save results in array
    case_pred={}
    case_pred={'Predicted':np.array([]),'Real':np.array([]),'Probability':np.array([]),'DecisionFunction':np.array([]),'Starttime':np.array([]),'Index':np.array([])}
    for i in Xn_background.T[-1].astype(int):#np.append(X_background.T[-1].astype(int),X_background.T[-1].astype(int)):

        case_pred['Predicted']=np.append(case_pred['Predicted'],y_pred[i])   
        case_pred['Real']=np.append(case_pred['Real'],y[i])
        case_pred['Probability']=np.append(case_pred['Probability'],y_proba[i][-1])
        case_pred['DecisionFunction']=np.append(case_pred['DecisionFunction'],dec[i])
        case_pred['Starttime']=np.append(case_pred['Starttime'],erg['Snippet']['Start Time'][i])
        case_pred['Index']=np.append(case_pred['Index'],i)
        for j,key in enumerate(train_keys):
            if not key in case_pred:
                case_pred[key]=np.array([X[i,j]])
            else:
                case_pred[key]=np.append(case_pred[key],X[i,j])

        for j,key in enumerate(background_keys):
            if not key in case_pred:
                case_pred[key]=np.array([X_background[i,j]])
            else:
                case_pred[key]=np.append(case_pred[key],X_background[i,j])   

    return case_pred
    
def plot_circulation_predictions(data,case_pred):
    signals,annot,shock_time=data
    roscs,rearrest=annot
    acctime,acc,ekgtime,ekg,co2time,co2=signals
    fig,ax=plt.subplots(3,figsize=(30,12),gridspec_kw={'height_ratios': [2,2,2]})
    for a in ax:
        for pos in roscs:
            a.axvline(pos,color='green',alpha=0.8,lw=3)
        for pos in rearrest:
            a.axvline(pos,color='orange',alpha=0.8,lw=3)
    ax[0].plot(acctime,0.01*(acc-np.mean(acc)),color='blue',alpha=0.3,label='ACC')
    ax[0].plot(ekgtime,ekg,color='green',alpha=0.3,label='EKG')
    ax[0].plot(shock_time,5*np.ones(len(shock_time)),ls='',marker='d',color='goldenrod',ms=10)
    ax[0].set_ylim(-10,10)
    ax[0].set_title('Acceleration and ECG')

    ax[1].plot(co2time,co2,color='goldenrod',alpha=0.9,label='Capnography')
    ax[1].set_ylim(-.1,60)
    ax[1].set_title('Capnography')


    ax[2].plot(case_pred['Starttime'],case_pred['Probability'],ls='',marker='.',color='indigo',alpha=0.7)
    t,f,std=mean_prob(case_pred['Starttime'],case_pred['Probability'],le=10,sigma=5)
    ax[2].plot(t,f,ls='-',marker='',color='purple',alpha=0.5,lw=3)
    ax[2].fill_between(t,f-std,f+std,color='violet',alpha=0.2)
    ax[2].plot(t,f+std,ls='-',marker='',color='violet',alpha=0.25,lw=1)
    ax[2].plot(t,f-std,ls='-',marker='',color='violet',alpha=0.25,lw=1)

    ax[2].set_ylim(-.1,1.1)
    ax[2].axhline(0.5,color='grey',alpha=0.7,lw=3)

    ax[2].set_title('SC Probability and Average Mean')
    # Legend Elements
    legend_elements0 = [Line2D([0], [0], color='blue', lw=2, label='Acceleration'),
                Line2D([0], [0], color='green', lw=2, label='ECG'),
                Line2D([0], [0], color='green', ls='-', label='ROSC',alpha=0.8,lw=3),
                Line2D([0], [0], color='orange', ls='-', label='Arrest',alpha=0.8,lw=3),
                Line2D([0], [0], color='goldenrod', marker='d',ls='', label='Shocks')       ]
    legend_elements1 = [Line2D([0], [0], color='goldenrod', lw=2, label='Capnography'),
                Line2D([0], [0], color='red', ls='-', label='ROSC',alpha=0.8,lw=3),
                Line2D([0], [0], color='orange', ls='-', label='Arrest',alpha=0.8,lw=3),]
    legend_elements2 = [Line2D([0], [0], color='indigo', marker='o',ls='', label='SC Probability'),
                Line2D([0], [0], color='purple', ls='-',lw=2, label='Rolling mean SC Probability'),
                Line2D([0], [0], color='violet', ls='-',lw=4, label='Standard deviation SC Probability',alpha=0.15),
                Line2D([0], [0], color='red', ls='-', label='ROSC',alpha=0.8,lw=3),
                Line2D([0], [0], color='orange', ls='-', label='Arrest',alpha=0.8,lw=3),]
    ax[0].legend(handles=legend_elements0,loc='upper left')
    ax[1].legend(handles=legend_elements1)
    ax[2].legend(handles=legend_elements2)

    plotendtime=acctime[-1]
    for a in ax:    
        a.grid()
        a.xaxis.set_ticks(np.arange(0,6000,100))
        a.set_xlim(acctime[0]-10,plotendtime+10)
    fig.show()
