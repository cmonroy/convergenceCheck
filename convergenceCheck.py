#!/usr/bin/env python
# -*- coding: latin_1 -*-

"""
  Convergence check of CFD simulations

 The numerical error is composed of the round-off error (neglected), the iterative error and the discretization error.

 Given that the residuals are properly converged, it is often assumed that the iterative error is small compared to the discretization error. 
 The discretization error consists in time and space (including surface representation - usually neglected) discretization erros. 
 There are different methods in order to evaluate the discretization error.

 The objective of this module is to implement these different methods.

# for unsteady computations, note that:
# - numerical results can consist of harmonic coefficients or min/max of the signals
# - the discretization error is the sum of time discretization error and spatial dicretization error
# - if we use the same ratio space over time in the different computations, the discretization error can be estimated only by spatial discretization error

 The scripts works with the pandas dataframe structure.

# Bibliography:
- Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications
I.B. Celik, U. Ghia, P.J. Roache, C.J. Freitas, H. Coleman
Journal of Fluids Engineering 130 (7)

"""

from __future__ import division #from python 3.0, / means floating division and // means floor division, while in python 2.7, / means floor division

import os
import sys
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt

import scipy.optimize as optimize


def GCI_3_simulations_and_constant_refinement_factor( df ):
#Convention in notation: 1 is finest, 3 is coarsest
   df.sort_values('dx',  inplace=True) #ascending=False,
   print df

   r=df['dx'].values[1]/df['dx'].values[0]
   #print r
   print df.columns
   print len(df.columns)

   extrapolated_21=np.zeros(len(df.columns)-3)
   print np.zeros(len(df.columns)-1)
   dfGCI= pd.DataFrame({'Simulation':["behaviour","extrapolated_value", "p", "GCI"]}  ,columns=df.columns)

   for icol, col in enumerate(df.columns):
     print df[col].name
     if ( df[col].name != 'Simulation' and df[col].name != 'dx' and df[col].name != 'dt' ):
        epsilon21 = df[col].values[1]-df[col].values[0]
        epsilon32 = df[col].values[2]-df[col].values[1]
        R= epsilon21/epsilon32 #R: discriminating ratio
        # Monotonic convergence: 0<R<1
        # Oscillatory convergence: R<0 and abs(R)<1
        # Monotonic divergence: R>1
        # Oscillatory divergence: R<0 and abs(R)>1
        if 0<R<1: #this method is only suited for monotonic convergence
           #p=np.log(1/R)/np.log(r)
           p=(1.0/np.log(r))*abs(np.log(abs(epsilon32/epsilon21)))
           extrapolated_21=(r**p*df[col].values[0]-df[col].values[1])/(r**p-1.0)
           GCI_21=1.25*abs(epsilon21/df[col].values[0])/(r**p-1.0)
           dfGCI[col].values[0]= "Monotonic convergence"
           dfGCI[col].values[1]= extrapolated_21
           dfGCI[col].values[2]= p
           dfGCI[col].values[3]= GCI_21

        elif ( R<0 and abs(R)<1 ):  #Proposed by Stern et al.
           U=0.5*(max(df[col].values)-min(df[col].values))
           dfGCI[col].values[2]= U

   dfConvergence=df.append(dfGCI, ignore_index=True);
   print dfConvergence

   return dfConvergence


def GCI( df ):
#based on Roache's original paper
#supported by ASME (terminology based on "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications")
   df.sort_values('dx',  inplace=True) #ascending=False,
   print df

   r21=df['dx'].values[1]/df['dx'].values[0]
   r32=df['dx'].values[2]/df['dx'].values[1]
   print df.columns
   print len(df.columns)

   extrapolated_21=np.zeros(len(df.columns)-3)
   print np.zeros(len(df.columns)-1)
   dfGCI= pd.DataFrame({'Simulation':["behaviour","extrapolated_value", "p", "GCI"]}  ,columns=df.columns)

   for icol, col in enumerate(df.columns):
     print df[col].name
     if ( df[col].name != 'Simulation' and df[col].name != 'dx' and df[col].name != 'dt' ):
        epsilon21 = df[col].values[1]-df[col].values[0]
        epsilon32 = df[col].values[2]-df[col].values[1]
        R= epsilon21/epsilon32 #R: discriminating ratio
        s=np.sign(epsilon32/epsilon21)
        # Monotonic convergence: 0<R<1
        # Oscillatory convergence: R<0 and abs(R)<1
        # Monotonic divergence: R>1
        # Oscillatory divergence: R<0 and abs(R)>1
        if (0<R<1):
           dfGCI[col].values[0]= "Monotonic convergence"
        elif (R<0 and abs(R)<1):
           dfGCI[col].values[0]= "Oscillatory convergence"
        elif (R>1):
           dfGCI[col].values[0]= "Monotonic divergence"
        else:
           dfGCI[col].values[0]= "Oscillatory divergence"
        if abs(R)<1: #this method is only suited for convergence
           def func(p):
               return 1/np.log(r21)*abs( np.log(abs(epsilon32/epsilon21))+np.log((r21**p-s)/(r32**p-s)))
           p=optimize.fixed_point(func,1/np.log(r21)*abs( np.log(abs(epsilon32/epsilon21))))
           extrapolated_21=(r21**p*df[col].values[0]-df[col].values[1])/(r21**p-1.0)
           GCI_21=1.25*abs(epsilon21/df[col].values[0])/(r21**p-1.0)
           dfGCI[col].values[1]= extrapolated_21
           dfGCI[col].values[2]= p
           dfGCI[col].values[3]= GCI_21

   dfConvergence=df.append(dfGCI, ignore_index=True);
   print dfConvergence

   return dfConvergence

   pass


def GCI_LS():
#based on the modified Roach's approach with least square fit
   pass

def checkDataFrame():
#check if input dataFrame is on good format (should include keys "Simulation", "dx", "dt" and at least one column of CFD results)
#make comments on steady/unsteady, Courant number...
   pass


def generateExampleInputFile(name):
   raw_data = {'Simulation': ["grid3", "grid2", "grid1"],
        'dx': [4.1,2,1],
        'dt': [6, 3, 1.5],
        'valueA': [1, 1.2, 1.3],
        'valueB': [3, 1, .5]}
   #df = pd.DataFrame(raw_data, columns = ['Simulation', 'dx', 'dt', 'valueA', 'valueB'])
   df = pd.DataFrame(raw_data)
   df.to_csv(name,index=False)




if __name__ == '__main__' :

   print ("Go")

   generateExampleInputFile('example.csv')
   df=pd.read_csv('example.csv')
   
   checkDataFrame()
   
#    plt.plot(df['dx'],df['valueA'], label='valueA')
#    plt.plot(df['dx'],df['valueB'], label='valueB')
#    plt.legend(loc='best')
#    plt.show()

   dfConvergence=GCI(df)
   dfConvergence.to_csv('exampleConvergence.csv')

   print ("Done")
