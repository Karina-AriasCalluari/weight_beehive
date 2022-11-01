# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:54:54 2022

@author: kd_be
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np
import scipy.optimize
from numpy import genfromtxt
from scipy import optimize
from numpy import vectorize
from scipy.optimize import curve_fit
from tabulate import tabulate
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

def f1(t,w,N_max, weight_bee,W_o,l):
    return  np.sum((w-(1*N_max*weight_bee+W_o-l*t))**2)
def myf1(t,N_max, weight_bee,W_o,l):
    w=(N_max*weight_bee+W_o-l*t)
    return w

def f2(t,w,a,d,m_food,N_max, weight_bee,W_o,l):
    return  np.sum((w-((a/(d+a)+d*np.exp(-(d+a)*t)/(d+a))*N_max*weight_bee+W_o-m_food*N_max*a*d/(d+a)**2+(m_food*N_max*a*(1-a/(d+a))-l)*t+m_food*N_max*a*d*np.exp(-(d+a)*t)/(d+a)**2))**2)

def myf2(t,a,d,m_food,N_max, weight_bee,W_o,l):
    w=((a/(d+a)+d*np.exp(-(d+a)*t)/(d+a))*N_max*weight_bee+W_o-m_food*N_max*a*d/(d+a)**2+(m_food*N_max*a*(1-a/(d+a))-l)*t+m_food*N_max*a*d*np.exp(-(d+a)*t)/(d+a)**2)
    return w

def f3(t,w,a,a2,d,m_food,N_max, weight_bee,F1,l):
    return  np.sum((w-((1-(1-a/(d+a))*np.exp(-a2*(t-1)))*N_max*weight_bee+F1+l+m_food*N_max*(1-a/(d+a))-l*t-m_food*N_max*(1-a/(d+a))*np.exp(-a2*(t-1))))**2)
def myf3(t,a,a2,d,m_food,N_max, weight_bee,F1,l):
    w=((1-(1-a/(d+a))*np.exp(-a2*(t-1)))*N_max*weight_bee+F1+l+m_food*N_max*(1-a/(d+a))-l*t-m_food*N_max*(1-a/(d+a))*np.exp(-a2*(t-1)))
    return w

def F1(a,d,m_food,N_max,W_o,l):
    return  (W_o-m_food*N_max*a*d/(d+a)**2+(m_food*N_max*a*(1-a/(d+a))-l)+m_food*N_max*a*d*np.exp(-(d+a))/(d+a)**2)

#x^{*}=(a/(d+a)+d*np.exp(-(d+a))/(d+a))
def Fw(t,a,a2,d,m_food,N_max, weight_bee,W_o,l):
    if (t < 0):
        w=(N_max*weight_bee+W_o-l*t)
    elif (t > 1):
        w=((1-(1-(a/(d+a)+d*np.exp(-(d+a))/(d+a)))*np.exp(-a2*(t-1)))*N_max*weight_bee+((W_o-m_food*N_max*a*d/(d+a)**2+(m_food*N_max*a*(1-a/(d+a))-l)+m_food*N_max*a*d*np.exp(-(d+a))/(d+a)**2) )+l+m_food*N_max*(1-(a/(d+a)+d*np.exp(-(d+a))/(d+a)))-l*t-m_food*N_max*(1-(a/(d+a)+d*np.exp(-(d+a))/(d+a)))*np.exp(-a2*(t-1)))
    else:
        w=((a/(d+a)+d*np.exp(-(d+a)*t)/(d+a))*N_max*weight_bee+W_o-m_food*N_max*a*d/(d+a)**2+(m_food*N_max*a*(1-a/(d+a))-l)*t+m_food*N_max*a*d*np.exp(-(d+a)*t)/(d+a)**2)
    return w
Wfunc = vectorize(Fw)

def Fx(t,a,a2,d):
    if   (t < 0):
          x= 1.0
    elif (t > 1):
          x=(1-(1-(a/(d+a)+d*np.exp(-(d+a))/(d+a)))*np.exp(-a2*(t-1)))
          #x=(1-(1-(a/(d+a)))*np.exp(-a2*(t-1)))
    else:
          x=a/(d+a)+d*np.exp(-(d+a)*t)/(d+a)
    return x
Fun= vectorize(Fx)

def Fxint(t_0,t_f,a,a2,d):
    return quad(Fx,t_0,t_f,args=(a,a2,d))[0]
vec_Fxint=np.vectorize(Fxint)

def fun_a2(a2, *data):
      t_0,t_1,a,d  = data   # automatic unpacking, no need for the 'i for i'
      return -vec_Fxint(0,1,a,a2,d)*d+vec_Fxint(0,1,a,a2,d)*(a2-a)+a+a2*((24-t_1)/(t_1-t_0))-a2*vec_Fxint(0,1+((24-t_1)/(t_1-t_0)),a,a2,d)
vec_fun_a2=np.vectorize(fun_a2)

def fun_a1(a1,*data):
    d,alpha,A,B,tc=data
    return (alpha*np.exp(tc*(d+a1))-(A-B)*d)/(A-B)-a1
#x^{*}=(a/(d+a)+d*np.exp(-(d+a))/(d+a))

#Defining m value in terms of B and $\alpha$
def fun_mf(mf, *data):
    A, B, alpha, N_max, a1, w,l = data   # automatic unpacking, no need for the 'i for i'
    return (mf*a1*N_max-alpha-l)/(a1**2*mf*N_max)+w/(mf*a1)+(B-A)/(alpha+l)


def fun_Tarrivals(t_0,t_1,a,a2,d,Nmax):
      return a*Nmax*(1-vec_Fxint(0,1,a,a2,d)) + a2*Nmax*((24-t_1)/(t_1-t_0))-a2*Nmax*vec_Fxint(1,1+(24-t_1)/(t_1-t_0),a,a2,d)
  
def fun_Tdepartures(a,a2,d,Nmax):
      return d*Nmax*vec_Fxint(0,1,a,a2,d)
  
    
def Line(x,m,b):
    y = m*x + b
    return y


def FilterErrors(T,W):
    diff_weight=np.diff(W)
    #Detecting indexes that exceeds 20 g
    indexes_2=np.where((abs(diff_weight)>=20)) 
    indexes= np.concatenate((indexes_2[0]-1,indexes_2[0]+1), axis=0) #Ojo 03/02/2021
    # Deleting indexes that exceeds 20 g
    T=np.delete(T,indexes)
    W=np.delete(W,indexes)
    return T,W

def FitTs(T,W,time_window=30):#,to,t1):
    #with open(filename,newline='') as filename:
        #my_data = genfromtxt(filename, delimiter=',')
        ##-> Update 15/11/2021 Outliers
        #T= (my_data[:,0])
        #W=(my_data[:,1])*1000
    diff_weight=np.diff(W)
        ##->Detecting indexes that exceeds 20 g
    indexes_2=np.where((abs(diff_weight)>=20)) 
    indexes= np.concatenate((indexes_2[0]-1,indexes_2[0],indexes_2[0]+1), axis=0) #Ojo 03/02/2021
        ##-> Deleting indexes that exceeds 20 g
    T=np.delete(T,indexes)
    W=np.delete(W,indexes)
        ##-> Update 15/11/2021 Outliers

    delta_W= np.diff(W)
    data_smooth=np.convolve(delta_W,np.ones(time_window,dtype=int),'valid')/time_window

        
    T_hat=T[int(time_window/2):-int(time_window/2)]
        

    result_1 = optimize.curve_fit(Line,np.concatenate((T_hat[:300],T_hat[-180:])),np.concatenate((data_smooth[:300],data_smooth[-180:])),p0=[0,1],bounds=([0,-1000], [0.0000001,1000]),maxfev=12000)
    result_2 = optimize.curve_fit(Line, T_hat[720:900], data_smooth[720:900],p0=[0,1],bounds=([0,-1000], [0.0000001,1000]),maxfev=12000)
    l_0=result_1[0][1]
    alpha_0=result_2[0][1]
        
    index_a=np.where((data_smooth==np.min(data_smooth)))[0]
    index_b=np.where((data_smooth==np.max(data_smooth)))[0]
        
    slopes=np.ones(len(T_hat))*l_0
    slopes[int(index_a[0]):int(index_b[0])]=alpha_0
        
        
        # From left to right:
    diff_2=(data_smooth-slopes)**2
    std=np.sqrt(np.cumsum(diff_2)/np.arange(start=1, stop=len(diff_2)+1))
        
        # From right to left:
    diff_2_i=np.flip(diff_2)
    std_i=np.sqrt(np.cumsum(diff_2_i)/np.arange(start=1, stop=len(diff_2)+1))
        

        # smooth from right to left:
    smooth = gaussian_filter1d(std_i, 10)#10
        # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))
        # find switching points
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
        
        
        # smooth from left to right:
    smooth_B = gaussian_filter1d(std, 100)#100
        # compute second derivative
    smooth_B2 = np.gradient(np.gradient(smooth_B))
        # find switching points
    infls_B = np.where(np.diff(np.sign(smooth_B2)))[0]
        
        
        
        ################  PRINTING PLOT  ################
    # fig=plt.figure(figsize=(12,7))
    # ax1 = fig.add_subplot(111)
        
    # ax1.plot( T_hat,data_smooth ,color ="#0000CD",linestyle='-')
    # ax1.plot(T_hat, std,color ="#CDB5CD",linestyle='-')
    # ax1.plot(T_hat, np.flip(smooth),color ="black",linestyle='-')
    # ax1.plot(T_hat, np.flip(std_i),color ="#9FB6CD",linestyle='-')
    # ax1.plot(T_hat, smooth_B,color ="#8A2BE2",linestyle='-')
        
    
    # ax1.vlines(x=[T_hat[int(index_a[0])], T_hat[int(index_b[0])]], ymin=[l_0, l_0], ymax=[alpha_0, alpha_0], colors='black', ls='--', lw=1, label='First Assumption')
    # ax1.hlines(y=[l_0,alpha_0,l_0], xmin=[T_hat[0], T_hat[int(index_a[0])],T_hat[int(index_b[0])]], xmax=[T_hat[int(index_a[0])],T_hat[int(index_b[0])], T_hat[-1]], colors='black', ls='-', lw=1)


    # ax1.set_ylabel('$\Delta$ Weight  [g]',fontname="Times New Roman",fontsize=20)
    # ax1.set_xlabel('t[min]',fontname="Times New Roman",fontsize=20)
            
    # ax1.tick_params(axis="y",direction="in", pad=7, length=13,labelsize=20)
    # ax1.tick_params(axis="x",direction="in", pad=7, length=13,labelsize=20)
        
        # for i, infl in enumerate(infls, 1):
        #     ax1.axvline(x=T_hat[-infl-1], color='#838B8B', label=f'Inflection Point {i}')
        
        # for i, infl in enumerate(infls_B, 1):
        #     ax1.axvline(x=T_hat[infl], color='red', label=f'Inflection Point {i}')
        
    Inf_l_r=T_hat[infls_B]
    val=np.min(abs(Inf_l_r-(T_hat[-1 ]/ 2)))
    index_m=np.where(abs(Inf_l_r-(T_hat[-1 ]/ 2))==val)
    t_0_l=Inf_l_r[int(index_m[0] - 1)]  #finding the t_0 from the left inflections
    t_1_l=Inf_l_r[int(index_m[0] + 1)]  #finding the t_0 from the left inflections
        


    Inf_l_r=T_hat[-infls-1]
    id_0_r=np.where(np.diff(np.sign(Inf_l_r-t_0_l))) # How to find the location of the first negative value for the average?
    id_1_r=np.where(np.diff(np.sign(Inf_l_r-t_1_l)))
        
    t_0_r=Inf_l_r[int(id_0_r[0] + 1)] #finding the t_0 from the right inflections
    t_1_r=Inf_l_r[int(id_1_r[0] + 1)] #finding the t_1 from the right inflections
        
        
    #ax1.axvline(x=t_0_r,color ="#9FB6CD", linestyle='-',linewidth=2.0)
    #ax1.axvline(x=t_1_r,color ="#9FB6CD", linestyle='-',linewidth=2.0)
        
    #ax1.axvline(x=t_0_l, color="#CDB5CD", linestyle='-',linewidth=2.5)
    #ax1.axvline(x=t_1_l, color="#CDB5CD", linestyle='-',linewidth=2.5)
    
    # ax1.axvspan(np.min([t_0_l,t_0_r]),np.max([t_0_l,t_0_r]), alpha=0.4, color="#DCDCDC")
    # ax1.axvspan(np.min([t_1_l,t_1_r]),np.max([t_1_l,t_1_r]), alpha=0.4, color="#DCDCDC")
        
    # ax1.axvline(x=(t_0_l+t_0_r)/(2), color='#CD0000', linestyle='--',linewidth=2.0,label ="$t_{0}$ $t_{1}$")
    # ax1.axvline(x=(t_1_l+t_1_r)/(2), color='#CD0000', linestyle='--',linewidth=2.0)
        
    # ax1.set_xlim([1,  1440])
    # ax1.legend(loc='lower left',fontsize=17)
        
    t_0=(t_0_l+t_0_r)/(2)
    t_1=(t_1_l+t_1_r)/(2)

    return  t_0,t_1#Inf_l_r[int(t_0_r[0] + 1)],Inf_l_r[int(t_1_r[0] + 1)]

#1.0 Previusly defined t0 and t1:

def Trescaled(T,to,t1):
    #2.0 Steps to rescale the time (hours:minutes) to integers
    time_hour=np.array([[to ,t1]])# Put the times here [7.7 , 17.1122449]
    delta_time = t1-to
    time_0=to/delta_time
    time_1=(24-t1)/delta_time
    interval=np.array(time_0+time_1+1)
    t = T/1440*interval-time_0
    return t

def RobustParameters_1(t,W,time_window=30): #Time rescaled!
    w_bee=0.113 #np.array([0.113]) #gr
    W_o=0 #We are joining F_o and W_o and considering as a W_t_0:

    weight=(W-W_o)#*1000
        
    ######## Estimation of W_0 and l:
    t_0_index= np.where(abs(t) == np.min(abs(t))) #Calculating the index for t=0
    t_1_index= np.where(abs(t-1) == np.min(abs(t-1))) #Calculating the index for t=1
    reg1 = LinearRegression().fit(t[:int(t_0_index[0])].reshape(-1, 1),weight[:int(t_0_index[0])])
    reg2 = LinearRegression().fit(t[-4*60:].reshape(-1, 1),weight[-4*60:])
        
    l_1=reg1.coef_[0]
    l_2=reg2.coef_[0]
    l_0=(l_1+l_2)/2
    A_=np.average(weight[:int(t_0_index[0])]-l_0*t[:int(t_0_index[0])])
    #print("A=",A_,"l=",l_0)
    
    delta_W=np.diff(weight)
    delta_t=np.diff(t)
    dW=delta_W/delta_t
    #time_window=30
    dW_smooth=np.convolve(dW,np.ones(time_window,dtype=int),'valid')/time_window
    T_dW_smooth=t[int(time_window/2):-int(time_window/2)]
        
    Index_dW_0= np.where(abs(T_dW_smooth) == np.min(abs(T_dW_smooth)))[0] #Calculating the index for t=0 in t vs dW/dt
    Index_dW_05= np.where(abs(T_dW_smooth-0.5) == np.min(abs(T_dW_smooth-0.5)))[0] #Calculating the index for t=0.5 in t vs dW/dt
        
    #Lets interpolate to find any accurate intersection with dW/dt=0
    Tc=[ ]
    for i in range(int(Index_dW_0),int(Index_dW_05)):
        x=T_dW_smooth[i:i+2]
        y=dW_smooth[i:i+2]
        #line = LinearRegression().fit(x.reshape(-1, 1),y)
        #b=line.intercept_
        #m=line.coef_[0]
            
        if np.sum(np.sign(y))==0:
            m=(dW_smooth[i+1]-dW_smooth[i])/(T_dW_smooth[i+1]-T_dW_smooth[i])
            popt,pcov = curve_fit(lambda x, _b: Line(x,m,_b), x, y)
            b=popt[0]
            #y_line=m*x+b
            data=(m,b)
            root=scipy.optimize.fsolve(Line,0.5, args=data)[0]
            #print("i=",i,"b=", popt[0], "m=", m,"tc=",root)
            Tc.append(root)
            #ax3.plot(x,y_line, linestyle='-',color ="red",linewidth=1.0)
    #print("tc=",Tc[0])
    t_c=[]
    W_t_c=[]
    for i in range(len(Tc)):
        t_c_index= np.where(abs(t-Tc[i]) == np.min(abs(t-Tc[i])))[0] #Calculating the index for tc
        #np.asarray()
        W_t_c.append(W[int(t_c_index)])
        t_c.append(t[int(t_c_index)])
    return l_0,A_,t_c,W_t_c


def RobustParameters_2(t,W,tc):
    W_o=0 #We are joining F_o and W_o and considering as a W_t_0:
    weight=(W-W_o)#*1000
    t_1_index= np.where(abs(t-1) == np.min(abs(t-1))) #Calculating the index for t=1
    t_c_index= np.where(abs(t-tc) == np.min(abs(t-tc))) #Calculating the index for tc
    reg3 = LinearRegression().fit(t[int(t_c_index[0]+7):int(t_1_index[0]-7)].reshape(-1, 1),weight[int(t_c_index[0]+7):int(t_1_index[0]-7)])
    alpha_=reg3.coef_[0]
    B_=reg3.intercept_
    return alpha_,B_

def fitParameters(t,W,to,t1,d,l,A,B,alpha,tc):
    #2.0 Steps to rescale the time (hours:minutes) to integers
    delta_time = t1-to
    time_0=to/delta_time
    time_1=(24-t1)/delta_time
    interval=np.array(time_0+time_1+1)
    W_o=0
    w_bee=0.113
    weight=(W-W_o)#*1000
    ######## Estimation of d (units):
    d=d*(t1-to)
    ######## Estimation of W_0 and l:
    # t_i_index= np.where(abs(t) == np.min(abs(t))) #Calculating the index for t=0
    # reg1 = LinearRegression().fit(t[:int(t_i_index[0])].reshape(-1, 1),weight[:int(t_i_index[0])])
    # #y_pred = reg.intercept_ + reg.coef_ * X
    # # print('intercept_1:', reg1.intercept_)
    # # print('slope_1:', reg1.coef_)
    # W_t0=reg1.intercept_
    # W_0= W_t0-w_bee*Nmax
    # l_0=reg1.coef_[0]
    #####  Running calibration model:(5 equations)
    data=(d,alpha,A,B,tc)
    a_1=scipy.optimize.fsolve(fun_a1,100, args=data)[0]
    N=alpha/(w_bee*d*np.exp(-tc*(d+a_1)))-(alpha+l)/(w_bee*d)
    m=(alpha+l)/N*(a_1+d)/(a_1*d)
    W0=A-w_bee*N
    data=(to,t1,a_1,d)
    a2_0=scipy.optimize.fsolve(fun_a2,17, args=data)[0]
    # Check the result by filtering in red the ones that exceed feasible values of our Table No 1 (Priors)
    result_4, pcov = optimize.curve_fit(Wfunc, t, weight,p0=[a_1,a2_0, d, m,N,w_bee,W0,l],bounds=([0.90*a_1,0.1*a2_0,0.9999*d,0.90*m ,0.90*N, w_bee*0.999999, W0*0.90,l*0.90], [1.1*a_1,2*a2_0,1.0000*d,1.1*m ,1.1*N, w_bee*1, W0*1.1,l*1.1]),maxfev=20000)
    #a_2=result_4[1]
    #a_2, pcov= optimize.curve_fit(lambda t, _a_2:vfunc(t,a_1,_a_2,d,m,N, w_bee,W0,l),t, weight,a2_0,maxfev=5000)
    #result_4=np.array([a_1,a_2[0], d, m,N,w_bee,W0,l])
    
    #3.0 Calculation of the error:
    Err_unit=np.sum((weight-Wfunc(t, *result_4))**2)
    #4.0 Ensambling results to plot
    Results_1=np.array([np.round(result_4[4]),result_4[6],result_4[0]/(t1-to),result_4[1]/(t1-to),result_4[2]/(t1-to),result_4[3],result_4[7]/(t1-to),Err_unit])
    Results_1=np.round(Results_1, 4)
    return Results_1

def Model_Estimation(filename,d,to,t1,tc=None,time_window=30):
    with open(filename,newline='') as filename:
        my_data = genfromtxt(filename, delimiter=',')
        # Update 15/11/2021 Outliers
        T= (my_data[:,0])
        W=(my_data[:,1])*1000
        diff_weight=np.diff(W)
        #Detecting indexes that exceeds 20 g
        indexes_2=np.where((abs(diff_weight)>=20)) 
        indexes= np.concatenate((indexes_2[0]-1,indexes_2[0]+1), axis=0) #Ojo 03/02/2021
        # Deleting indexes that exceeds 20 g
        T=np.delete(T,indexes)
        W=np.delete(W,indexes)
        # Update 15/11/2021 Outliers
        
        time_hour=np.array([[to ,t1]])# Put the times here [7.7 , 17.1122449]
        w_bee=0.113 #np.array([0.113]) #gr
        W_o=0 #We are joining F_o and W_o and considering as a W_t_0:
        Results_1=np.zeros([len(time_hour),8])
        
        #2.0 Steps to rescale the time (hours:minutes) to integers
        delta_time = t1-to
        time_0=to/delta_time
        time_1=(24-t1)/delta_time
        interval=np.array(time_0+time_1+1)

        t = T/1440*interval-time_0
        weight=(W-W_o)#*1000
        
        ######## Estimation of t (indexes):
        
        t_0_index= np.where(abs(t) == np.min(abs(t))) #Calculating the index for t=0
        t_1_index= np.where(abs(t-1) == np.min(abs(t-1))) #Calculating the index for t=1
        
        ######## Rescaling d term ########:
        d=d*(t1-to)
        ######## Estimation of W_0 and l:
        reg1 = LinearRegression().fit(t[:int(t_0_index[0])].reshape(-1, 1),weight[:int(t_0_index[0])])
        reg1 = LinearRegression().fit(t[:int(t_0_index[0])].reshape(-1, 1),weight[:int(t_0_index[0])])
        reg2 = LinearRegression().fit(t[-4*60:].reshape(-1, 1),weight[-4*60:])
        
        l_1=reg1.coef_[0]
        l_2=reg2.coef_[0]
        l_0=(l_1+l_2)/2
        A =np.average(weight[:int(t_0_index[0])]-l_0*t[:int(t_0_index[0])])
        
        ######## Estimation of t_c:
        if tc==None:
            delta_W=np.diff(weight)
            delta_t=np.diff(t)
            dW=delta_W/delta_t
            #time_window=30
            dW_smooth=np.convolve(dW,np.ones(time_window,dtype=int),'valid')/time_window
            T_dW_smooth=t[int(time_window/2):-int(time_window/2)]
        
            Index_dW_0= np.where(abs(T_dW_smooth) == np.min(abs(T_dW_smooth)))[0] #Calculating the index for t=0 in t vs dW/dt
            Index_dW_05= np.where(abs(T_dW_smooth-0.5) == np.min(abs(T_dW_smooth-0.5)))[0] #Calculating the index for t=0.5 in t vs dW/dt
            Index_dW_min= np.where(dW_smooth == np.min(dW_smooth))[0] #Calculating the index for dW/dt=min in t vs dW/dt
            #Lets interpolate to find any accurate intersection with dW/dt=0
            Tc=[ ]
            for i in range(int(Index_dW_0),int(Index_dW_05)):
                x=T_dW_smooth[i:i+2]
                y=dW_smooth[i:i+2]
                if np.sum(np.sign(y))==0:
                    m=(dW_smooth[i+1]-dW_smooth[i])/(T_dW_smooth[i+1]-T_dW_smooth[i])
                    popt,pcov = curve_fit(lambda x, _b: Line(x,m,_b), x, y)
                    b=popt[0]
                    data=(m,b)
                    root=scipy.optimize.fsolve(Line,0.5, args=data)[0]
                    Tc.append(root)
        
            #print(T_dW_smooth[int(Index_dW_min)])
            Tc=np.concatenate([[T_dW_smooth[int(Index_dW_min)]],Tc])
            tc_attempt=np.linspace(Tc[0],Tc[1],30)
            #tc_attempt=np.linspace(0.1,Tc[0],40)
            Error=[ ]
            Results=[ ]
            for i in range(len(tc_attempt)):
                ######## Estimation of variables:
                    tc=tc_attempt[i]
                    t_c_index= np.where(abs(t-tc) == np.min(abs(t-tc))) #Calculating the index for tc
                    reg3 = LinearRegression().fit(t[int(t_c_index[0]+7):int(t_1_index[0]-7)].reshape(-1, 1),weight[int(t_c_index[0]+7):int(t_1_index[0]-7)])
                    alpha=reg3.coef_[0]
                    B=reg3.intercept_
                    data=(d,alpha,A,B,tc)
                    a_1=scipy.optimize.fsolve(fun_a1,100, args=data)[0]
                    N=alpha/(w_bee*d*np.exp(-tc*(d+a_1)))-(alpha+-l_0)/(w_bee*d)
                    m=(alpha+-l_0)/N*(a_1+d)/(a_1*d)
                    W0=A-w_bee*N
                    data=(to,t1,a_1,d)
                    a2_0=scipy.optimize.fsolve(fun_a2,17, args=data)[0]
                    # Check the result by filtering in red the ones that exceed feasible values of our Table No 1 (Priors)
                    result, pcov = optimize.curve_fit(Wfunc, t, weight,p0=[a_1,a2_0, d, m,N,w_bee,W0,-l_0],bounds=([0.90*a_1,0.1*a2_0,0.9999*d,0.90*m ,0.90*N, w_bee*0.999999, W0*0.90,-l_0*0.90], [1.1*a_1,2*a2_0,1.0000*d,1.1*m ,1.1*N, w_bee*1, W0*1.1,-l_0*1.1]),maxfev=20000)
                    Err_unit=np.sum((W-Wfunc(t, *result))**2)/len(weight)
                    Error.append(Err_unit)
                    Results.append(result)
            
            I=Error.index(min(Error))
            result=Results[I]
            error=Error[I]
        else:
            tc=(tc-to)/(t1-to)
            t_c_index= np.where(abs(t-tc) == np.min(abs(t-tc))) #Calculating the index for tc
            reg3 = LinearRegression().fit(t[int(t_c_index[0]+7):int(t_1_index[0]-7)].reshape(-1, 1),weight[int(t_c_index[0]+7):int(t_1_index[0]-7)])
            alpha=reg3.coef_[0]
            B=reg3.intercept_
            data=(d,alpha,A,B,tc)
            a_1=scipy.optimize.fsolve(fun_a1,100, args=data)[0]
            N=alpha/(w_bee*d*np.exp(-tc*(d+a_1)))-(alpha+-l_0)/(w_bee*d)
            m=(alpha+-l_0)/N*(a_1+d)/(a_1*d)
            W0=A-w_bee*N
            data=(to,t1,a_1,d)
            a2_0=scipy.optimize.fsolve(fun_a2,17, args=data)[0]
            # Check the result by filtering in red the ones that exceed feasible values of our Table No 1 (Priors)
            result, pcov = optimize.curve_fit(Wfunc, t, weight,p0=[a_1,a2_0, d, m,N,w_bee,W0,-l_0],bounds=([0.90*a_1,0.1*a2_0,0.9999*d,0.90*m ,0.90*N, w_bee*0.999999, W0*0.90,-l_0*0.90], [1.1*a_1,2*a2_0,1.0000*d,1.1*m ,1.1*N, w_bee*1, W0*1.1,-l_0*1.1]),maxfev=20000)
            error=np.sum((W-Wfunc(t, *result))**2)/len(weight)

        ################  PRINTING PLOT  ################
        fig=plt.figure(figsize=(12,7))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
            
        ax1.plot(t, Wfunc(t, *result) ,color ="#990000", linestyle='-',linewidth=3.0)
        ax1.scatter(t,weight,color ="#0000CD",s=13)#linewidth=3.7)
   
        ax1.legend([ "$W(t)$","$W'(t)$"], loc ="lower right",fontsize=17)
        ax1.set_ylabel('Weight  [g]',fontname="Times New Roman",fontsize=17)
        ax1.set_xlabel('t',fontname="Times New Roman",fontsize=17)
            
        ax1.tick_params(axis="y",direction="in", pad=7, length=13,labelsize=17)
        ax1.tick_params(axis="x",direction="in", pad=7, length=13,labelsize=17)

        def tick_function(X):
            V = (X+time_0)/interval * 1440/60
            return ["%.2f" % z for z in V]
            
        hours=np.array([0.0, 4. , 8. , 12. ,16. , 20. ,24.])
        new_tick_locations=(hours*60*interval/1440)-time_0
            
        ax2.set_xlim(ax1.get_xlim()) #ax2
        ax2.set_xticks(new_tick_locations) #ax2
        ax2.set_xticklabels(tick_function(new_tick_locations)) #ax2
        ax2.tick_params(axis="x",direction="out", pad=18, length=13,labelsize=17) #ax2
        ax2.set_xlabel("t [h]",fontname="Times New Roman",fontsize=17) #ax2
            
        #plt.savefig('Hive_G.eps', format='eps', dpi=1000)
    col_names = ["N[bees]","W0[g]","a[1/h]","a2[1/h]","d[1/h]","m[g]","l[g/h]","Error$^2$"]
    #4.0 Ensambling results to plot
    Results_1[0,:]=[round(result[4]),result[6],result[0]/(t1-to),result[1]/(t1-to),result[2]/(t1-to),result[3],result[7]/(t1-to),error]
    Results_1=np.round(Results_1, 4)
    print(tabulate(Results_1, headers=col_names, tablefmt='fancy_grid'))
    #print(W_t0,l_0,l_0/(t1-to))
    return Results_1

        #1.0 Previusly defined t0 and t1:
def Model_Estimation_2(filename,to,t1,Nmax,m_fo=0.01,a_0=10,a2_0=20,d_0=10):
    with open(filename,newline='') as filename:
        my_data = genfromtxt(filename, delimiter=',')
        # Update 15/11/2021 Outliers
        T= (my_data[:,0])
        W=(my_data[:,1])*1000
        diff_weight=np.diff(W)
        #Detecting indexes that exceeds 20 g
        indexes_2=np.where((abs(diff_weight)>=20)) 
        indexes= np.concatenate((indexes_2[0]-1,indexes_2[0]+1), axis=0) #Ojo 03/02/2021
        # Deleting indexes that exceeds 20 g
        T=np.delete(T,indexes)
        W=np.delete(W,indexes)
        # Update 15/11/2021 Outliers
        
        time_hour=np.array([[to ,t1]])# Put the times here [7.7 , 17.1122449]
        w_bee=0.113 #np.array([0.113]) #gr
        W_o=0 #We are joining F_o and W_o and considering as a W_t_0:
        Results_1=np.zeros([len(time_hour),8])
        
        #2.0 Steps to rescale the time (hours:minutes) to integers
        delta_time = t1-to
        time_0=to/delta_time
        time_1=(24-t1)/delta_time
        interval=np.array(time_0+time_1+1)

        t = T/1440*interval-time_0
        weight=(W-W_o)#*1000
        
        ######## Estimation of W_0 and l:
        t_i_index= np.where(abs(t) == np.min(abs(t))) #Calculating the index for t=0
        reg1 = LinearRegression().fit(t[:int(t_i_index[0])].reshape(-1, 1),weight[:int(t_i_index[0])])
        #y_pred = reg.intercept_ + reg.coef_ * X
        # print('intercept_1:', reg1.intercept_)
        # print('slope_1:', reg1.coef_)
        W_t0=reg1.intercept_
        W_0= W_t0-w_bee*Nmax
        l_0=reg1.coef_[0]
        #####  Running calibration model:
        # Check the results_4 by filtering in red the ones that exceed feasible values of our Table No 1 (Priors)
        result_4, pcov = optimize.curve_fit(Wfunc, t,  weight,p0=[a_0,a2_0, d_0, m_fo,Nmax,w_bee,W_0,-l_0],bounds=([0,0,0,0 ,Nmax*0.99999, w_bee*0.9999, W_0-1000,0], [max(weight)/w_bee,max(weight)/w_bee,max(weight)/w_bee,w_bee,Nmax, w_bee,W_0+1000, max(weight)]),maxfev=12000)
        perr = np.sqrt(np.diag(pcov))
            
        B=result_4[0]*result_4[4]*(1-result_4[0]/(result_4[2]+result_4[0]))*result_4[3]
        tc=np.log((B-result_4[7])/(result_4[5]*result_4[4]*result_4[2]+B))/-(result_4[2]+result_4[0]) 

        #3.0 Calculation of the error:
        Err_unit=np.sum((weight-Wfunc(t, *result_4))**2)/len(weight)
          
        #4.0 Ensambling results to plot
        Results_1[0,:]=[round(result_4[4]),result_4[6],result_4[0]/(t1-to),result_4[1]/(t1-to),result_4[2]/(t1-to),result_4[3],result_4[7]/(t1-to),Err_unit]

        ################  PRINTING PLOT  ################
        fig=plt.figure(figsize=(12,7))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
            
        ax1.plot(t, Wfunc(t, *result_4) ,color ="#990000", linestyle='-',linewidth=3.0)
        ax1.scatter(t,weight,color ="#0000CD",s=13)#linewidth=3.7)
   
        ax1.legend([ "$W(t)$","$W'(t)$"], loc ="lower right",fontsize=17)
        ax1.set_ylabel('Weight  [g]',fontname="Times New Roman",fontsize=17)
        ax1.set_xlabel('t',fontname="Times New Roman",fontsize=17)
            
        ax1.tick_params(axis="y",direction="in", pad=7, length=13,labelsize=17)
        ax1.tick_params(axis="x",direction="in", pad=7, length=13,labelsize=17)

        def tick_function(X):
            V = (X+time_0)/interval * 1440/60
            return ["%.2f" % z for z in V]
            
        hours=np.array([0.0, 4. , 8. , 12. ,16. , 20. ,24.])
        new_tick_locations=(hours*60*interval/1440)-time_0
            
        ax2.set_xlim(ax1.get_xlim()) #ax2
        ax2.set_xticks(new_tick_locations) #ax2
        ax2.set_xticklabels(tick_function(new_tick_locations)) #ax2
        ax2.tick_params(axis="x",direction="out", pad=18, length=13,labelsize=17) #ax2
        ax2.set_xlabel("t [h]",fontname="Times New Roman",fontsize=17) #ax2
            
        #plt.savefig('Hive_G.eps', format='eps', dpi=1000)
    col_names = ["N[bees]","W0[g]","a[1/h]","a2[1/h]","d[1/h]","m[g]","l[g/h]","Error$^2$"]
    
    Results_1=np.round(Results_1, 3)
    print(tabulate(Results_1, headers=col_names, tablefmt='fancy_grid'))
    return Results_1#.T#perr,t,weight,pcov,

#r=Model_Estimation('Hive_1.csv',0.854,8.90,17.77,9.521)
#r=Model_Estimation('Hive_2.csv',0.85,8.30,17.36,9.598)
#r=Model_Estimation('Hive_3.csv',0.854,8.70,18.42,9.90) #:)
#r=Model_Estimation('Hive_4.csv',0.854,8.20,18.26,9.59)
#r=Model_Estimation('Hive_5.csv',0.854,8.10,17.36,9.565)
#r=Model_Estimation('Hive_6.csv',0.854,8.40,17.93,9.74) #:)
#r=Model_Estimation('Hive_7.csv',0.854,8.40,17.93,10.525) #:)
#r=Model_Estimation('Hive_9.csv',0.85,8.40,17.77,9.95)
#r=Model_Estimation('Hive_10.csv',0.854,8.10,17.93,11.42)

# def my_fitting(filename,to,t1,Nmax,m_fo=0.01,a_0=10,a2_0=20,d_0=10):
#     with open(filename,newline='') as filename:
#         my_data = genfromtxt(filename, delimiter=',')
#         # Update 15/11/2021 Outliers
#         T= (my_data[:,0])
#         W=(my_data[:,1])*1000
#         diff_weight=np.diff(W)
#         #Detecting indexes that exceeds 20 g
#         indexes_2=np.where((abs(diff_weight)>=20)) 
#         indexes= np.concatenate((indexes_2[0]-1,indexes_2[0]+1), axis=0) #Ojo 03/02/2021
#         # Deleting indexes that exceeds 20 g
#         T=np.delete(T,indexes)
#         W=np.delete(W,indexes)
#         # Update 15/11/2021 Outliers
        
#         time_hour=np.array([[to ,t1]])# Put the times here [7.7 , 17.1122449]
#         w_bee=0.113 #np.array([0.113]) #gr
#         W_o=0 #We are joining F_o and W_o and considering as a W_t_0:
#         Results_1=np.zeros([len(time_hour),8])
        
#         #2.0 Steps to rescale the time (hours:minutes) to integers
#         delta_time = t1-to
#         time_0=to/delta_time
#         time_1=(24-t1)/delta_time
#         interval=np.array(time_0+time_1+1)

#         t = T/1440*interval-time_0
#         weight=(W-W_o)#*1000
        
#         ######## Estimation of W_0 and l:
#         t_i_index= np.where(abs(t) == np.min(abs(t))) #Calculating the index for t=0
#         reg1 = LinearRegression().fit(t[:int(t_i_index[0])].reshape(-1, 1),weight[:int(t_i_index[0])])
#         #y_pred = reg.intercept_ + reg.coef_ * X
#         # print('intercept_1:', reg1.intercept_)
#         # print('slope_1:', reg1.coef_)
#         W_t0=reg1.intercept_
#         W_0= W_t0-w_bee*Nmax
#         l_0=reg1.coef_[0]
#         #####  Running calibration model:
        
#         result_4, pcov = optimize.curve_fit(Wfunc, t,  weight,p0=[a_0,a2_0, d_0, m_fo,Nmax,w_bee,W_0,-l_0],bounds=([0,0,0,0 ,Nmax*0.99999, w_bee*0.9999, W_0-1000,0], [max(weight)/w_bee,max(weight)/w_bee,max(weight)/w_bee,w_bee,Nmax, w_bee,W_0+1000, max(weight)]),maxfev=12000)
#         perr = np.sqrt(np.diag(pcov))
            
#         B=result_4[0]*result_4[4]*(1-result_4[0]/(result_4[2]+result_4[0]))*result_4[3]
#         tc=np.log((B-result_4[7])/(result_4[5]*result_4[4]*result_4[2]+B))/-(result_4[2]+result_4[0]) 

#         #3.0 Calculation of the error:
#         Err_unit=np.sum((weight-Wfunc(t, *result_4))**2)
          
#         #4.0 Ensambling results to plot
#         Results_1[0,:]=[round(result_4[4]),result_4[6],result_4[0]/(t1-to),result_4[1]/(t1-to),result_4[2]/(t1-to),result_4[3],result_4[7]/(t1-to),Err_unit]

#         ################  PRINTING PLOT  ################
#         fig=plt.figure(figsize=(12,7))
#         ax1 = fig.add_subplot(111)
#         ax2 = ax1.twiny()
            
#         ax1.plot(t, Wfunc(t, *result_4) ,color ="#990000", linestyle='-',linewidth=3.0)
#         ax1.scatter(t,weight,color ="#0000CD",s=13)#linewidth=3.7)
   
#         ax1.legend([ "$W(t)$","$W'(t)$"], loc ="lower right",fontsize=17)
#         ax1.set_ylabel('Weight  [g]',fontname="Times New Roman",fontsize=17)
#         ax1.set_xlabel('t',fontname="Times New Roman",fontsize=17)
            
#         ax1.tick_params(axis="y",direction="in", pad=7, length=13,labelsize=17)
#         ax1.tick_params(axis="x",direction="in", pad=7, length=13,labelsize=17)

#         def tick_function(X):
#             V = (X+time_0)/interval * 1440/60
#             return ["%.2f" % z for z in V]
            
#         hours=np.array([0.0, 4. , 8. , 12. ,16. , 20. ,24.])
#         new_tick_locations=(hours*60*interval/1440)-time_0
            
#         ax2.set_xlim(ax1.get_xlim()) #ax2
#         ax2.set_xticks(new_tick_locations) #ax2
#         ax2.set_xticklabels(tick_function(new_tick_locations)) #ax2
#         ax2.tick_params(axis="x",direction="out", pad=18, length=13,labelsize=17) #ax2
#         ax2.set_xlabel("t [h]",fontname="Times New Roman",fontsize=17) #ax2
            
#         #plt.savefig('Hive_G.eps', format='eps', dpi=1000)
#     col_names = ["N[bees]","W0[g]","a[1/h]","a2[1/h]","d[1/h]","m[g]","l[g/h]","Error$^2$"]
    
#     Results_1=np.round(Results_1, 3)
#     print(tabulate(Results_1, headers=col_names, tablefmt='fancy_grid'))
#     return Results_1#.T#perr,t,weight,pcov,

# https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python

