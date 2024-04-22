# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:56:31 2023

@author: RAM1COT
"""

"""
Created on Tue Oct 31 15:50:36 2023

@author: RAM1COT
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:56:31 2023

@author: RAM1COT
"""

import numpy as np
import sympy as sym
from sympy import MatrixSymbol, Matrix
import matplotlib.pyplot as plt

#%% Material parameters, Geometric parameters, Loading 

mu=0.3
E= (169e9/(1-mu**2))*np.array([[1,mu,0],[mu,1,0],[0,0,0.5*(1-mu)]])
Lhorizontal=3
Lvertical=2
T= 1e-6
F= 100

#%% Mesh
nHorizontal = 8
nVertical = 8
nElem= nHorizontal*nVertical                                  
nNodes = (nVertical+1) * (nHorizontal+1)                               # for 4 noded elements in 2D
meshH= Lhorizontal/nHorizontal
meshV= Lvertical/nVertical
x=np.zeros(((nHorizontal+1)*(nVertical+1),1))
y=np.zeros(((nHorizontal+1)*(nVertical+1),1))   

#%% Boundary conditions

# Force BC
nodeF=[2*nNodes-1]
Fvec=np.zeros((2*nNodes,1))
Fvec[nodeF]=[F]

# Displacement BC
nodeU= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]                                                      # Location of the fixed nodal points
Up=np.zeros((2*nNodes,1)) 
Up[2*nNodes-1]=0                            

#%%  Displacement and strain interpolation matrix

r=sym.Symbol('r')
s=sym.Symbol('s')
def interpolation(r,s,X1,X2,X3,X4,Y1,Y2,Y3,Y4):
    
    
    h4 = 0.25*(1+r)*(1-s)
    h3 = 0.25*(1-r)*(1-s)
    h2 = 0.25*(1+r)*(1+s)
    h1 = 0.25*(1-r)*(1+s)

    h4dr = 0.25*(1-s) 
    h3dr = -0.25*(1-s)
    h2dr = 0.25*(1+s)
    h1dr = -0.25*(1+s)
    
    h4ds = -0.25*(1+r) 
    h3ds = -0.25*(1-r)
    h2ds = 0.25*(1+r)
    h1ds = 0.25*(1-r)
    
    J1=MatrixSymbol('J1',2,2)
    J=Matrix(J1)
    J[0,0] = h1dr*X1 + h2dr*X2 + h3dr*X3 + h4dr*X4
    J[0,1] = h1dr*Y1 + h2dr*Y2 + h3dr*Y3 + h4dr*Y4
    J[1,0] = h1ds*X1 + h2ds*X2 + h3ds*X3 + h4ds*X4
    J[1,1] = h1ds*Y1 + h2ds*Y2 + h3ds*Y3 + h4ds*Y4   
    detJ   = J.det()
    JI     = J.inv()
    
    H1dr = JI[0,0]*h1dr + JI[0,1]*h1ds
    H2dr = JI[0,0]*h2dr + JI[0,1]*h2ds
    H3dr = JI[0,0]*h3dr + JI[0,1]*h3ds
    H4dr = JI[0,0]*h4dr + JI[0,1]*h4ds
    
    H1ds = JI[1,0]*h1dr + JI[1,1]*h1ds
    H2ds = JI[1,0]*h2dr + JI[1,1]*h2ds
    H3ds = JI[1,0]*h3dr + JI[1,1]*h3ds
    H4ds = JI[1,0]*h4dr + JI[1,1]*h4ds
       
    B1=MatrixSymbol('B1',8,3)
    B = Matrix(B1) 
    B[:,0] = [H1dr,0,H2dr,0,H3dr,0,H4dr,0]
    B[:,1] = [0,H1ds,0,H2ds,0,H3ds,0,H4ds]
    B[:,2] = [H1ds,H1dr,H2ds,H2dr,H3ds,H3dr,H4ds,H4dr]
    Bmatrix = B.T
    
    return Bmatrix, detJ, J


#%% Stiffness Matrix

K_init = np.zeros((nElem,2*nNodes,2*nNodes))    
k = np.zeros((nElem,8,8)) 
                                                                    # Global stiffness matrix (tensor) initialization
def Kmatrix():
    nCount = 0 
                                                  # local stiffness matrix for 3 noded 1D element
    for elemV in range(nVertical):
        for elemH in range(nHorizontal):
            
            nCount += 1
            S1 = 2*nCount - 2 + 2*(elemV) 
            S2 = S1+1
            S3 = S1+2
            S4 = S1+3
            S5 = 2*(nCount+nHorizontal) - 2 + 2*(elemV+1)
            S6 = S5+1
            S7 = S5+2
            S8 = S5+3                                               # 1st node of each element in the Global K matrix
            S=np.array([S1,S2,S3,S4,S5,S6,S7,S8])
            
            X1= (elemH)*meshH
            X2= (elemH+1)*meshH 
            X3= X1
            X4= X2
            
            Y1= -(elemV)*meshV
            Y2= Y1
            Y3= -(elemV+1)*meshV
            Y4= Y3
            
            pt1 = int(S1/2)
            pt2 = int(S3/2)
            pt3 = int(S5/2)
            pt4 = int(S7/2)
            x[pt1]=X1
            x[pt2]=X2
            x[pt3]=X3
            x[pt4]=X4
            y[pt1]=Y1
            y[pt2]=Y2
            y[pt3]=Y3
            y[pt4]=Y4
            
            
            Bmatrix,detJ,J=interpolation(r,s,X1,X2,X3,X4,Y1,Y2,Y3,Y4)
            BCBt=np.matmul( np.transpose(Bmatrix) , np.matmul(E,Bmatrix) )
            for jj in range(8):
                for kk in range(8):
                    k[nCount-1,jj,kk] = T*sym.integrate(BCBt[jj,kk]*detJ, (r,-1,1), (s,-1,1) )  
                    K_init[nCount-1,S[jj],S[kk]] = k[nCount-1,jj,kk].copy() 
    return K_init, k                         
K_init, k  = Kmatrix() 
K=sum(K_init[ii,:,:] for ii in range(nElem))  
   
#%% Corrected system (Enforcing Displacement BC)

def CorrectedSystem(K1):
    for ii in range(len(nodeU)):
        Kr1=np.delete(K1,nodeU[ii],axis=0)   
        Kr2=np.delete(Kr1,nodeU[ii],axis=1)  
        Kr3=np.insert(Kr2,nodeU[ii],np.zeros((1,2*nNodes-1)),axis=0) 
        Kr=np.insert(Kr3,nodeU[ii],np.zeros((1,2*nNodes)),axis=1)
        Kr[nodeU[ii],nodeU[ii]]=1
        K1=Kr
    return Kr
Kr = CorrectedSystem(K)

#%%  Solution

f =  Fvec - np.matmul(K,Up) 
K_inv=np.linalg.inv(Kr)     
U=np.dot(K_inv,f)
U[nodeU]=Up[nodeU]

#%% Plotting

Ux=np.zeros((nHorizontal+1,nVertical+1))
Uy=np.zeros((nHorizontal+1,nVertical+1))
X=np.zeros((nHorizontal+1,nVertical+1))
Y=np.zeros((nHorizontal+1,nVertical+1))
counter=0
for ii in range(nHorizontal +1):
    for jj in range(nVertical+1):
        Ux[ii,jj] = U[counter]
        Uy[ii,jj] = U[counter+1]
        counter += 2
        
plt.imshow(Ux)
plt.legend(["X deflection"])  
plt.colorbar()
plt.show()
 
plt.imshow(Uy)
plt.legend(["Y deflection"])  
plt.colorbar()
plt.show()

#%%
nElements = [1,4,9,16,25,36,49,64]
myCode= [.0013,.00195,.00243,.00279,.00307,.0033,.00349,.00387]
comsol = [.0022,.0032,.0037,.004,.0043,.0045,.0047,.0049]
plt.figure()
plt.plot(nElements,myCode,'b--')
plt.plot(nElements,comsol,'r--')
plt.xlabel('Number of elements')
plt.ylabel( 'Dispalcement at the point of application of force (m)')
plt.legend(["My Code (element used : 4 noded rectangle)", "COMSOL (element used : 8 noded rectangle)"])