#!/usr/bin/env python
# -*- coding: latin_1 -*-

"""
Convergence check of CFD simulations

#Author: Charles Monroy

#License: GNU GPL

#Description:
The numerical error is composed of the round-off error (neglected), the iterative error and the discretization error.

Given that the residuals are properly converged, it is often assumed that the iterative error is small compared to the discretization error.
The discretization error consists in time and space (including surface representation - usually neglected) discretization erros.
There are different methods in order to evaluate the discretization error. The objective of this module is to implement these different methods.

For unsteady computations, note that:
- numerical results can consist of harmonic coefficients or min/max of the signals
- the discretization error is the sum of time discretization error and spatial dicretization error
- if we use the same ratio space over time in the different computations, the discretization error can be estimated only by spatial discretization error

The scripts work with the pandas dataframe structure.

# Bibliography:
[1] "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications"
I.B. Celik, U. Ghia, P.J. Roache, C.J. Freitas, H. Coleman, Journal of Fluids Engineering 130 (7)
freely accessible at http://fluidsengineering.asmedigitalcollection.asme.org/article.aspx?articleid=1434171

[2] "Quantitative V&V of CFD simulations and certification of CFD codes"
F. Stern, R. Wilson and J. Shao, Int. J. Numer. Meth. Fluids 2006; 50:1335–1355

[3] "Verification of Solutions in Unsteady Flows"
Eca, Hoekstra & Vaz, ASME 2015

[4] "Robust regression" by Patrich Breheny
freely accessible at http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf

[5] "Robust Regression" by John Fox & Sanford Weisberg
freely accesible at http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf
"""

from __future__ import division #from python 3.0, / means floating division and // means floor division, while in python 2.7, / means floor division

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as optimize


def GCI_3_simulations_and_constant_refinement_factor( df ):
#Convention in notation: 1 is finest, 3 is coarsest
   df.sort_values('dx',  inplace=True)

   r=df['dx'].values[1]/df['dx'].values[0]

   extrapolated_21=np.zeros(len(df.columns)-3)
   dfGCI= pd.DataFrame({'Simulation':["behaviour","extrapolated_value", "p", "GCI"]}  ,columns=df.columns)

   for icol, col in enumerate(df.columns):
     if ( df[col].name != 'Simulation' and df[col].name != 'dx' and df[col].name != 'dt' ):
        epsilon21 = df[col].values[1]-df[col].values[0]
        epsilon32 = df[col].values[2]-df[col].values[1]
        R= epsilon21/epsilon32 #R: discriminating ratio
        if 0<R<1: #the method proposed in [1] is only suited for monotonic convergence
           p=(1.0/np.log(r))*abs(np.log(abs(epsilon32/epsilon21)))
           extrapolated_21=(r**p*df[col].values[0]-df[col].values[1])/(r**p-1.0)
           GCI_21=1.25*abs(epsilon21/df[col].values[0])/(r**p-1.0)
           dfGCI[col].values[0]= "Monotonic convergence"
           dfGCI[col].values[1]= extrapolated_21
           dfGCI[col].values[2]= p
           dfGCI[col].values[3]= GCI_21

        elif ( R<0 and abs(R)<1 ):  #Proposed by [2]
           U=0.5*(max(df[col].values)-min(df[col].values))
           dfGCI[col].values[2]= U

   dfConvergence=df.append(dfGCI, ignore_index=True);

   return dfConvergence


def GCI( df ):
#supported by ASME (name of variables based on [1])
   df.sort_values('dx',  inplace=True) #ascending=False,

   r21=df['dx'].values[1]/df['dx'].values[0]
   r32=df['dx'].values[2]/df['dx'].values[1]

   extrapolated_21=np.zeros(len(df.columns)-3)
   dfGCI= pd.DataFrame({'Simulation':["behaviour","extrapolated_value", "p", "GCI"]}  ,columns=df.columns)

   for icol, col in enumerate(df.columns):
     if ( df[col].name != 'Simulation' and df[col].name != 'dx' and df[col].name != 'dt' ):
        epsilon21 = df[col].values[1]-df[col].values[0]
        epsilon32 = df[col].values[2]-df[col].values[1]
        R= epsilon21/epsilon32 #R: discriminating ratio
        s=np.sign(epsilon32/epsilon21)
        if (0<R<1):
           dfGCI[col].values[0]= "Monotonic convergence"
        elif (R<0 and abs(R)<1):
           dfGCI[col].values[0]= "Oscillatory convergence"
        elif (R>1):
           dfGCI[col].values[0]= "Monotonic divergence"
        else:
           dfGCI[col].values[0]= "Oscillatory divergence"
        if abs(R)<1: #the method proposed in [1] is only suited for monotonic convergence
           def func(p):
               return 1/np.log(r21)*abs( np.log(abs(epsilon32/epsilon21))+np.log((r21**p-s)/(r32**p-s)))
           p=optimize.fixed_point(func,1/np.log(r21)*abs( np.log(abs(epsilon32/epsilon21))))
           extrapolated_21=(r21**p*df[col].values[0]-df[col].values[1])/(r21**p-1.0)
           GCI_21=1.25*abs(epsilon21/df[col].values[0])/(r21**p-1.0)
           dfGCI[col].values[1]= extrapolated_21
           dfGCI[col].values[2]= p
           dfGCI[col].values[3]= GCI_21
   dfConvergence=df.append(dfGCI, ignore_index=True);
   return dfConvergence

def GCI_LS():
#based on the modified Roach's approach with least square fit
   pass


def unsteady_convergence(df, lossFunction = "Tukey"):
#based on presentation of "Verification of Solutions in Unsteady Flows" by  Eca, Hoekstra & Vaz at ASME 2015

   def SRE_LS(x, h, tau, phi):
      """
        Least square minimization
      """
      SRE=0.0
      for i in range(0,len(h)-1):
         SRE=SRE+(phi[i]-(x[0]+x[1]*h[i]**x[2]+x[3]*tau[i]**x[4]))**2
      return SRE

   def SRE_method2(x, h, tau, phi):
      """
        Second weithing function proposed by Eca et al.
      """
      SRE=0.0
   
      totalWeigth=0
      for i in range(0,len(h)-1):
         totalWeigth=totalWeigth+(1/h[i]+1/tau[i])
   
      for i in range(0,len(h)-1):
         wi=(1/h[i]+1/tau[i])/totalWeigth
         SRE=SRE+wi*(phi[i]-(x[0]+x[1]*h[i]**x[2]+x[3]*tau[i]**x[4]))**2
      return SRE

   def func(x, h, tau):
      func=x[0]+x[1]*h**x[2]+x[3]*tau**x[4]
      return func

   def SRE_Huber(x, h, tau, phi):
      """
       Huber estimator
   
       "Huber argued that c = 1.345 is a good choice, and showed
       that asymptotically, it is 95% as efficient as least squares if
       the true distribution is normal (and much more efficient in
       many other cases)", see [4]
   
       see [5] for algorithm
      """
      
      c=1.345
      sigma=np.std(func(x, h, tau)-phi)
      k=c*sigma

      SRE=0.0
      for i in range(0,len(h)-1):
         fi= x[0]+x[1]*h[i]**x[2]+x[3]*tau[i]**x[4]
         ri= phi[i] - fi
         if abs(ri)<k:
            wi=1
            SRE=SRE+wi*0.5*ri**2
         else:
            wi=k/abs(ri)
            SRE=SRE+ wi*(k*abs(ri)-0.5*k**2)
      return SRE


   def SRE_Tukey(x, h, tau, phi):
      """
       Tukey estimator, also known as bisquare estimator

       "The value c = 4.685 is usually used for this loss function,
       and again, it provides an asymptotic efficiency 95% that of linear regression
       for the normal distribution", see [4]
       
       see [5] for algorithm
      """

      c=4.685
      sigma=np.std(func(x, h, tau)-phi)
      k=c*sigma

      SRE=0.0
      for i in range(0,len(h)-1):
         fi= x[1]+x[0]*h[i]**x[2]+x[3]*tau[i]**x[4]
         ri= phi[i] - fi
         if abs(ri)<k:
            wi=(1-(ri/c)**2)**2
            SRE=SRE+wi*((k**2)/6)*(1-(1-(ri/k)**2)**3)
         else:
             SRE=SRE+ 0
      return SRE



   dicoReader = {
                "LS" : SRE_LS ,
                "method2" : SRE_method2 ,
		"Huber" : SRE_Huber ,
		"Tukey" : SRE_Tukey ,
   		 }


   df.sort_values('dx',  inplace=True)
   df.sort_values('dt',  inplace=True)

   df_Unsteady= pd.DataFrame({'Simulation':["behaviour","extrapolated_value", "alpha_x", "p_x", "alpha_t", "p_t"]}  ,columns=df.columns)
   df_Unsteady["dx"].values[1]= 0.0
   df_Unsteady["dt"].values[1]= 0.0
   for icol, col in enumerate(df.columns):
     print df[col].name
     if ( df[col].name != 'Simulation' and df[col].name != 'dx' and df[col].name != 'dt' ):

         #x0 = np.zeros(5, dtype = float) #x=[phi0,alpha_x,p_x,alpha_t,p_t]
         x0 = np.array([df[col].values[0],1.0,1.5,1.0,1.5]) #x=[phi0,alpha_x,p_x,alpha_t,p_t]

         res = optimize.fmin_powell(dicoReader[lossFunction], x0, args=(df['dx'], df['dt'], df[col]), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=1, disp=1, retall=0, callback=None, direc=None)

         df_Unsteady[col].values[1]= res[0][0] #phi_0
         df_Unsteady[col].values[2]= res[0][1] #alpha_x
         df_Unsteady[col].values[3]= res[0][2] #p_x
         df_Unsteady[col].values[4]= res[0][3] #alpha_t
         df_Unsteady[col].values[5]= res[0][4] #p_t

         if ((0.5 < df_Unsteady[col].values[3] < 2) and (0.5 < df_Unsteady[col].values[5] < 2 )):
             df_Unsteady[col].values[0]="normal"
         else:
             df_Unsteady[col].values[0]="anomalous"

   dfConvergence=df.append(df_Unsteady, ignore_index=True);
   return dfConvergence

def checkDataFrame():
#check if input dataFrame is on good format (should include keys "Simulation", "dx", "dt" and at least one column of CFD results)
#make comments on steady/unsteady, Courant number...
   pass


def generateExampleInputFile1D(name):
   raw_data = {'Simulation': ["grid3", "grid2", "grid1"],
        'dx': [4.1,2,1],
        'dt': [6, 3, 1.5],
        'valueA': [1, 1.2, 1.3],
        'valueB': [3, 1, .5]}
   #df = pd.DataFrame(raw_data, columns = ['Simulation', 'dx', 'dt', 'valueA', 'valueB'])
   df = pd.DataFrame(raw_data)
   df.to_csv(name,index=False)


def generateExampleInputFile2D(name):
   raw_data = {'Simulation': ["sim1", "sim2", "sim3", "sim4", "sim5", "sim6", "sim7", "sim8"],
        'dx': [1,1,1,0.5,0.5,0.5,0.385,0.385],
        'dt': [0.075,0.01,0.0075,0.075,0.01,0.0075,0.075,0.01],
        'VBM_max': [1668434000,1635725000,1639637000,1650991000,1819359000,1847338000,1590012000,1980035000],
        'VBM_min': [-4611240000,-4819855000,-4763167000,-4743176000,-5108901000,-5158990000,-4848864000,-5199893000]}
   df = pd.DataFrame(raw_data)
   df.to_csv(name,index=False)

if __name__ == '__main__' :
   from mpl_toolkits.mplot3d import Axes3D
   import matplotlib.pyplot as plt
   print ("Go")


   df=generateExampleInputFile2D('example2D.csv')
   df=pd.read_csv('example2D.csv')
   print df

   dfConvergence=unsteady_convergence(df, lossFunction="LS")

   print dfConvergence
   dfConvergence.to_csv('exampleConvergence2D.csv')

   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot_trisurf(df['dx'], df['dt'], df['VBM_max'], cmap=plt.cm.Spectral, linewidth=0.2, antialiased=True)
   ax.scatter(df['dx'], df['dt'],  df['VBM_max'], zdir='z', s=20, c=None, depthshade=True)
   ax.set_xlabel('dx')
   ax.set_ylabel('dt')
   ax.set_zlabel('VBM_max')
   plt.show()
   plt.close()





#    generateExampleInputFile1D('example1D.csv')
#    df=pd.read_csv('example.csv')
# 
#    checkDataFrame()
# 
#    plt.plot(df['dx'],df['valueA'], label='valueA')
#    plt.plot(df['dx'],df['valueB'], label='valueB')
#    plt.legend(loc='best')
#    plt.show()
# 
#    dfConvergence1D=GCI(df)
# 
#    print dfConvergence1D
# 
#    dfConvergence1D.to_csv('exampleConvergence1D.csv')

   print ("Done")
