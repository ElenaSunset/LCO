# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:18:23 2021

@author: Elena
"""

#LiCoO2 R-3m Structures


import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.offsetbox import AnchoredText
from PIL import Image
from mpl_toolkits import mplot3d
import math
import itertools
from matplotlib.patches import FancyArrowPatch




##############################################################################
#Real Space Structure Definition
##############################################################################

#Hexagonal Axes
ahx=2.84
bhx=2.84
chx=14.14
#Real Hexagonal Lattice 
xh1=ahx/2
yh1=-ahx*math.sqrt(3)/2
zh1=0
xh2=ahx/2
yh2=ahx*math.sqrt(3)/2
zh2=0
xh3=0
yh3=0
zh3=chx
ah1=np.array([xh1,yh1,zh1])
ah2=np.array([xh2,yh2,zh2])
ah3=np.array([xh3,yh3,zh3])
#Real Space Vectors Hexagonal Cell
ah=pd.DataFrame({"x":[0,xh1,xh2,xh3],"y":[0,yh1,yh2,yh3],"z":[0,zh1,zh2,zh3]})



#Rhombohedral Axes
arh=4.98185
brh=4.98185
crh=4.98185
#Real Rhombohedral Lattice
alphar=32.9997*np.pi/180
betar=32.9997*np.pi/180
gammar=32.9997*np.pi/180
xr1=0;
yr1=ahx*math.sqrt(3)/3
zr1=chx/3
xr2=ahx/2
yr2=-ahx*math.sqrt(3)/6
zr2=chx/3
xr3=-ahx/2
yr3=-ahx*math.sqrt(3)/6
zr3=chx/3
ar1=np.array([xr1,yr1,zr1])
ar2=np.array([xr2,yr2,zr2])
ar3=np.array([xr3,yr3,zr3])
#Real Space Vectors Rhombohedral Cell
ar=pd.DataFrame({"x":[0,xr1,xr2,xr3],"y":[0,yr1,yr2,yr3],"z":[0,zr1,zr2,zr3]})
Da111=ar1+ar2+ar3

#Real Space Cells
fig1 = plt.figure(1)
ax1=plt.axes(projection='3d')
ax1.plot(ar.loc[[0,1]].x,ar.loc[[0,1]].y,ar.loc[[0,1]].z,label="a",color="g")
ax1.plot(ar.loc[[0,2]].x,ar.loc[[0,2]].y,ar.loc[[0,2]].z,label="b",color="b")
ax1.plot(ar.loc[[0,3]].x,ar.loc[[0,3]].y,ar.loc[[0,3]].z,label="c",color="r")
ax1.plot([0,Da111[0]],[0,Da111[1]],[0,Da111[2]],label="[111] real",color="k")  
ax1.plot(ah.loc[[0,1]].x,ah.loc[[0,1]].y,ah.loc[[0,1]].z,label="a hex",color="g",linestyle=":")
ax1.plot(ah.loc[[0,2]].x,ah.loc[[0,2]].y,ah.loc[[0,2]].z,label="b hex",color="b",linestyle=":")
ax1.plot(ah.loc[[0,3]].x,ah.loc[[0,3]].y,ah.loc[[0,3]].z,label="c hex",color="r",linestyle=":")
ax1.legend()
plt.title("Hex vs Rhomb axes")
plt.show()



##############################################################################
#Reciprocal Space Definition
##############################################################################

#Reciprocal Rhombohedral Lattice
ar1xar2=np.cross(ar1,ar2)
ar2xar3=np.cross(ar2,ar3)
ar3xar1=np.cross(ar3,ar1)
Vr=np.dot(ar1,ar2xar3)
#reciprocal vectors
br1=2*np.pi*(ar2xar3)/Vr
br2=2*np.pi*(ar3xar1)/Vr
br3=2*np.pi*(ar1xar2)/Vr
br=pd.DataFrame({"x":[0,br1[0],br2[0],br3[0]],"y":[0,br1[1],br2[1],br3[1]],"z":[0,br1[2],br2[2],br3[2]]})
Db111=br1+br2+br3


#Reciprocal Hexagonal Lattice
ah1xah2=np.cross(ah1,ah2)
ah2xah3=np.cross(ah2,ah3)
ah3xah1=np.cross(ah3,ah1)
Vh=np.dot(ah1,ah2xah3)
#reciprocal vectors
bh1=2*np.pi*(ah2xah3)/Vh
bh2=2*np.pi*(ah3xah1)/Vh
bh3=2*np.pi*(ah1xah2)/Vh
bh=pd.DataFrame({"x":[0,bh1[0],bh2[0],bh3[0]],"y":[0,bh1[1],bh2[1],bh3[1]],"z":[0,bh1[2],bh2[2],bh3[2]]})

#Reciprocal Symmetry vectors
eta=(1+4*math.cos(alphar))/(2+4*math.cos(alphar))
nu=3/4 - eta/2


#################################

#Reciprocal Space Symmetry Points 
G=np.array([0,0,0]) #Gamma Point
#Rombohedral
Z0=1/2*br1 + 1/2*br2 + 1/2*br3 #Surface Gamma T in Bilbao Notation
Tr=Z0
Lr1=1/2*br1 #3fold M Volume
Lr_1=-1/2*br1
Lr2=1/2*br2 
Lr_2=-1/2*br2
Lr3=1/2*br3 
Lr_3=-1/2*br3

#Surface K or "B"
B0=eta*br1+1/2*br2+(1-eta)*br3 #Surface K
B1=np.array([-0.69302639, 1.20035692, 0.66653309])
B2=np.array([-1.385405340045676, 0, 0.6670048096793617])
B3=np.array([1.385405340045676, 0, 0.6670048096793617])
B4=np.array([-0.69302639, -1.20035692, 0.66653309])
B5=np.array([0.69302639, -1.20035692, 0.66653309])
#bottom face
B_0=np.array([0.69302639, 1.20035692, -0.66653309])
B_1=np.array([-0.69302639, 1.20035692, -0.66653309])
B_2=np.array([-1.385405340045676, 0, -0.6670048096793617])
B_3=np.array([1.385405340045676, 0, -0.6670048096793617])
B_4=np.array([-0.69302639, -1.20035692, -0.66653309])
B_5=np.array([0.69302639, -1.20035692, -0.66653309])

BrPts=pd.DataFrame({"x":[B0[0],B1[0],B2[0],B3[0],B4[0],B5[0],B_0[0],B_1[0],B_2[0],B_3[0],B_4[0],B_5[0]],"y":[B0[1],B1[1],B2[1],B3[1],B4[1],B5[1],B_0[1],B_1[1],B_2[1],B_3[1],B_4[1],B_5[1]],"z":[B0[2],B1[2],B2[2],B3[2],B4[2],B5[2],B_0[2],B_1[2],B_2[2],B_3[2],B_4[2],B_5[2]]})

H1=1/2*br1 + (1-eta)*br2 + (eta-1)*br3 #H in Bilbao Notation
H2=np.array([-0.826,1.277,0.22])
H3=np.array([0.826,-1.277,-0.22])
H4=np.array([-0.826,-1.277,-0.22])
H5=np.array([0.693,1.354,-0.22])
H6=np.array([-0.693,1.354,-0.22])
H7=np.array([-0.693,-1.354,0.22])
H8=np.array([0.693,-1.354,0.22])
H9=np.array([1.54,0,0.22])
H10=np.array([-1.5,0.1,0.22])
H11=np.array([1.5,-0.1,-0.22])
H12=np.array([-1.5,-0.1,-0.22])

F1=1/2*br1+1/2*br2 #center of square faces
F2=1/2*br2+1/2*br3
F3=1/2*br1+1/2*br3

P=eta*br1+nu*br2+nu*br3 #Surface M 
P_2=nu*br1+eta*br2+nu*br3 #my guess of position
P_3=nu*br1+nu*br2+eta*br3 #my guess of position
P1=(1-nu)*br1 + (1-nu)*br2 + (1-eta)*br3 #surface Mprime
#L1=-1/2*br3 #3fold Mprime Volume
P_4=(1-eta)*br1+(1-nu)*br2+(1-nu)*br3#my guess
P_5=(1-nu)*br1+(1-eta)*br2+(1-nu)*br3#my guess

P2=nu*br1+nu*br2+(eta-1)*br3 #M in volume
X=nu*br1-nu*br3 #M' in volume

#Adjacent Cells
G1=br1 #next BZs centers G
G2=br2
G3=br3
G4=-br1
G5=-br2
G6=-br3
G5=br1+br2+br3
G6=G1+(br1+br2+br3)
G_6=G1-(br1+br2+br3)
G7=G2 + (br1+br2+br3)
G_7=G2 - (br1+br2+br3)
G8=G3 + (br1+br2+br3)
G_8=G3 - (br1+br2+br3)
G9=-(br1+br2+br3)
GrPts=pd.DataFrame({"x":[G[0],G1[0],G2[0],G3[0],G4[0],G5[0],G6[0],G7[0],G8[0],G9[0],G_6[0],G_7[0],G_8[0]],"y":[G[1],G1[1],G2[1],G3[1],G4[1],G5[1],G6[1],G7[1],G8[1],G9[1],G_6[1],G_7[1],G_8[1]],"z":[G[2],G1[2],G2[2],G3[2],G4[2],G5[2],G6[2],G7[2],G8[2],G9[2],G_6[2],G_7[2],G_8[2]]})

Z0=1/2*(br1+br2+br3)
Z1=G1 + (1/2*br1+1/2*br2+1/2*br3) #tipoM
Z2=G2 + (1/2*br1+1/2*br2+1/2*br3) #tipoM
Z3=G3 + (1/2*br1+1/2*br2+1/2*br3) #tipoM
Z4=G4 + (1/2*br1+1/2*br2+1/2*br3) #tipo M'
Z5=G5 + (1/2*br1+1/2*br2+1/2*br3)#tipo M
Z6=G6 + (1/2*br1+1/2*br2+1/2*br3)#tipo M'
Z7=G7 + (1/2*br1+1/2*br2+1/2*br3)
ZrPts=pd.DataFrame({"x":[Z0[0],Z1[0],Z2[0],Z3[0],Z4[0],Z5[0],Z6[0],Z7[0]],"y":[Z0[1],Z1[1],Z2[1],Z3[1],Z4[1],Z5[1],Z6[1],Z7[1]],"z":[Z0[2],Z1[2],Z2[2],Z3[2],Z4[2],Z5[2],Z6[2],Z7[2]]})



###################################################
#Translation to other notations:
#M points are P points in other notations
M0=P
M1=P1
M2=P_2
M3=P_3
M4=P_4
M5=P_5

#Adjacent Cells (Hexagonal non primitive cell) These are NOT REAL Symmetry points
Gh1=bh1
Gh2=bh2
Gh3=bh3
Gh4=Gh3+bh3
Gh5=Gh4+bh3
Gh6=Gh5+bh3
Gh7=2*bh1
Gh8=2*bh2
Gh9=-bh3

Zh0=bh3/2
Zh1=bh1+bh3/2
Zh2=bh2+bh3/2
Zh3=bh3+bh3/2
Zh4=-bh3/2

#Hexaonal surface Ms
# Mh=1/2*bh2 + bh3
# Mh2=1/2*bh1 + bh3
#Equivalent to T lower symmetry (non primitive)
Ah=1/2*bh3
#Equivalent to volume M/L
Lh1=-1/2*bh1+1/2*bh2+1/2*bh3
Lh2=1/2*bh1-1/2*bh2-1/2*bh3
Lh3=1/2*bh1+1/2*bh2+1/2*bh3  
#Kh=1/3*bh1+1/3*bh2
Hh=1/3*bh1+1/3*bh2+1/2*bh3
Kh0=1/3*bh1+1/3*bh2+1/2*bh3
Mph0=1/2*bh2 + bh3/2 #including manually the 3-fold symmetry
Mh0=1/2*bh1 + bh3/2



fig2=plt.figure(2)
ax2=plt.subplot()
ax2=plt.axes(projection="3d")
ax2.plot(ar.loc[[0,1]].x,ar.loc[[0,1]].y,ar.loc[[0,1]].z,label="Rhombohedral Real Axes",color="r")
ax2.plot(ar.loc[[0,2]].x,ar.loc[[0,2]].y,ar.loc[[0,2]].z,color="r")
ax2.plot(ar.loc[[0,3]].x,ar.loc[[0,3]].y,ar.loc[[0,3]].z,color="r")
ax2.plot([0,Da111[0]],[0,Da111[1]],[0,Da111[2]],label="[111] real",color="r",linestyle="--",alpha=0.5)
ax2.plot(br.loc[[0,1]].x,br.loc[[0,1]].y,br.loc[[0,1]].z,label="Rhombohedral Reciprocal Axes",linewidth=5,linestyle="-",color="orange")
ax2.plot(br.loc[[0,2]].x,br.loc[[0,2]].y,br.loc[[0,2]].z,linestyle="-",color="orange",linewidth=5)
ax2.plot(br.loc[[0,3]].x,br.loc[[0,3]].y,br.loc[[0,3]].z,linestyle="-",color="orange",linewidth=5)
ax2.plot([0,Db111[0]],[0,Db111[1]],[0,Db111[2]],label="[111] reciprocal",color="orange",linestyle="--",alpha=0.5)
ax2.plot(ah.loc[[0,1]].x,ah.loc[[0,1]].y,ah.loc[[0,1]].z,label="Hexagonal Real Axes",color="b",alpha=0.5)
ax2.plot(ah.loc[[0,2]].x,ah.loc[[0,2]].y,ah.loc[[0,2]].z,color="b",alpha=0.5)
ax2.plot(ah.loc[[0,3]].x,ah.loc[[0,3]].y,ah.loc[[0,3]].z,color="b",alpha=0.5)
ax2.plot(bh.loc[[0,1]].x,bh.loc[[0,1]].y,bh.loc[[0,1]].z,label="Hexagonal Reciprocal Axes",color="c",linewidth=5)
ax2.plot(bh.loc[[0,2]].x,bh.loc[[0,2]].y,bh.loc[[0,2]].z,color="c",linewidth=5)
ax2.plot(bh.loc[[0,3]].x,bh.loc[[0,3]].y,bh.loc[[0,3]].z,color="c",linewidth=5)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.set_aspect('auto')    
# set_axes_equal(ax2)
plt.title("Rhombohedral real and reciprocal axes")
ax2.legend()
plt.show()




################################################################################################
#Equivalencies to other notations - Theoretical Points and vectors
################################################################################################
#Symmetry point positions, THeoretical Materialscloud.org *bi_t* (rotation of axes with respect to my previous definition)
#Note that there is a 30º or 60 rotation from one notation to the other and therefore points and vectors are not equivalent
#A comparison can be made between ar dataframe and a_t (fig 3)

#Real R-3m 166
a1_t=np.array([1.4180062517,0.8186862911,4.7165839515])
a2_t=np.array([-1.4180062517,0.8186862911,4.7165839515])
a3_t=np.array([0,-1.6373725822,4.7165839515])
a_t=pd.DataFrame({"x":[0,a1_t[0],a2_t[0],a3_t[0]],"y":[0,a1_t[1],a2_t[1],a3_t[1]],"z":[0,a1_t[2],a2_t[2],a3_t[2]]})

Li_t= 1.5*a1_t + 0.5*a2_t + 0.5*a3_t
Co_t= 0*a1_t + 0*a2_t + 0*a3_t
O_t= 0.739587*a1_t + 0.739587*a2_t + 0.739587*a3_t
O2_t= 1.2604130000*a1_t + 1.2604130000*a2_t + 0.2604130000*a3_t


#Reciprocal R-3m 166
b1_t=np.array([2.2154998611,1.2791194412,0.4440491517])
b2_t=np.array([-2.2154998611,1.2791194412,0.4440491517])
b3_t=np.array([0,-2.5582388824,0.4440491517])
b_t=pd.DataFrame({"x":[0,b1_t[0],b2_t[0],b3_t[0]],"y":[0,b1_t[1],b2_t[1],b3_t[1]],"z":[0,b1_t[2],b2_t[2],b3_t[2]]})


#Symmetry Points defined in new notation:
F_t=0.5*b1_t + 0.5*b3_t
F2_t=0.5*b1_t + 0.5*b2_t

G_t=0*b1_t + 0*b2_t + 0*b3_t

T_t= 0.5*b1_t + 0.5*b2_t + 0.5*b3_t

H0_t=0.5*b1_t - 0.1867524438*b2_t + 0.1867524438*b3_t
H2_t=0.8132475562*b1_t + 0.1867524438*b2_t + 0.5*b3_t
H4_t=0.8132475562*b1_t + 0.5*b2_t + 0.1867524438*b3_t
H6_t=0.5*b1_t + 0.1867524438*b2_t - 0.1867524438*b3_t

L_t= 0.5*b1_t + 0*b2_t + 0*b3_t
L2_t= 0*b1_t  - 0.5*b2_t + 0*b3_t
L4_t= 0*b1_t + 0*b2_t - 0.5*b3_t

M0_t= 0.3433762219*b1_t - 0.1867524438*b2_t + 0.3433762219*b3_t
M2_t= 0.6566237781*b1_t + 0.1867524438*b2_t + 0.6566237781*b3_t
M4_t= 0.8132475562*b1_t + 0.3433762219*b2_t + 0.3433762219*b3_t
M6_t= 0.6566237781*b1_t + 0.6566237781*b2_t + 0.1867524438*b3_t
M8_t= 0.3433762219*b1_t + 0.3433762219*b2_t - 0.1867524438*b3_t

S0_t= 0.3433762219*b1_t - 0.3433762219*b2_t + 0*b3_t
S2_t= 0.6566237781*b1_t + 0*b2_t + 0.3433762219*b3_t
S4_t= 0.3433762219*b1_t + 0*b2_t - 0.3433762219*b3_t
S6_t= 0.6566237781*b1_t + 0.3433762219*b2_t + 0*b3_t



fig3=plt.figure(3)
ax3=plt.subplot()
ax3=plt.axes(projection="3d",proj_type = 'ortho')

ax3.plot(ar.loc[[0,1]].x,ar.loc[[0,1]].y,ar.loc[[0,1]].z,label="real notation 1",color="k",linestyle="-")
ax3.plot(ar.loc[[0,2]].x,ar.loc[[0,2]].y,ar.loc[[0,2]].z,color="k",linestyle="-")
ax3.plot(ar.loc[[0,3]].x,ar.loc[[0,3]].y,ar.loc[[0,3]].z,color="k",linestyle="-")
ax3.plot(br.loc[[0,1]].x,br.loc[[0,1]].y,br.loc[[0,1]].z,label="reciprocal notation 1",color="k",linestyle=":")
ax3.plot(br.loc[[0,2]].x,br.loc[[0,2]].y,br.loc[[0,2]].z,color="k",linestyle=":")
ax3.plot(br.loc[[0,3]].x,br.loc[[0,3]].y,br.loc[[0,3]].z,color="k",linestyle=":")

ax3.plot(a_t.loc[[0,1]].x,a_t.loc[[0,1]].y,a_t.loc[[0,1]].z,label="real notation 2",color="r",linestyle="-")
ax3.plot(a_t.loc[[0,2]].x,a_t.loc[[0,2]].y,a_t.loc[[0,2]].z,color="r",linestyle="-")
ax3.plot(a_t.loc[[0,3]].x,a_t.loc[[0,3]].y,a_t.loc[[0,3]].z,color="r",linestyle="-")
ax3.plot(b_t.loc[[0,1]].x,b_t.loc[[0,1]].y,b_t.loc[[0,1]].z,label="reciprocal notation 2",color="r",linestyle=":")
ax3.plot(b_t.loc[[0,2]].x,b_t.loc[[0,2]].y,b_t.loc[[0,2]].z,color="r",linestyle=":")
ax3.plot(b_t.loc[[0,3]].x,b_t.loc[[0,3]].y,b_t.loc[[0,3]].z,color="r",linestyle=":")

ax3.legend()
# ax3.set_aspect('equal')
# set_axes_equal(ax3)
plt.show()


D1 = np.linalg.norm(np.array(M0_t[1])-np.array(M0_t[2]))
D2 = np.linalg.norm(np.array(M0_t[0])-np.array(M0_t[1]))
D3 = np.linalg.norm(np.array(M0_t[0])-np.array(G[0]))
a1_t.tolist()
a2_t.tolist()
a3_t.tolist()

################################################################################
#Following figure shows both reciprocal cells and the most important symmetry points in the First BZ and adjacent BZs

fig4=plt.figure(4)
ax4=plt.subplot()
ax4=plt.axes(projection="3d",proj_type = 'ortho')
#Plotting the reciprocal space vectors
#Rhombohedral reciprocal vectors
ax4.plot(br.loc[[0,1]].x,br.loc[[0,1]].y,br.loc[[0,1]].z,label="Rhombohedral Reciprocal Axes",linestyle=":",color="r")
ax4.plot(br.loc[[0,2]].x,br.loc[[0,2]].y,br.loc[[0,2]].z,linestyle=":",color="r")
ax4.plot(br.loc[[0,3]].x,br.loc[[0,3]].y,br.loc[[0,3]].z,linestyle=":",color="r")
#Hexagonal reciprocal vectors
ax4.plot(bh.loc[[0,1]].x,bh.loc[[0,1]].y,bh.loc[[0,1]].z,label="hexagonal Reciprocal Axes",linestyle=":",color="k")
ax4.plot(bh.loc[[0,2]].x,bh.loc[[0,2]].y,bh.loc[[0,2]].z,linestyle=":",color="k")
ax4.plot(bh.loc[[0,3]].x,bh.loc[[0,3]].y,bh.loc[[0,3]].z,linestyle=":",color="k")
#directiin 111 in Rhombohedral axes (001 in hexaogonal)
ax4.plot([0,Db111[0]],[0,Db111[1]],[0,Db111[2]],linestyle="--",label="[111] reciprocal",color="y")

#Plotting the Brillouin zones using the defined points:
#111 Face of Rhombohedral Brillouin Zone
ax4.plot(BrPts.loc[[0,1]].x,BrPts.loc[[0,1]].y,BrPts.loc[[0,1]].z,linestyle="-",color="k",label="Rhombohedral BZ")
ax4.plot(BrPts.loc[[2,1]].x,BrPts.loc[[2,1]].y,BrPts.loc[[2,1]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[2,4]].x,BrPts.loc[[2,4]].y,BrPts.loc[[2,4]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[5,4]].x,BrPts.loc[[5,4]].y,BrPts.loc[[5,4]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[5,3]].x,BrPts.loc[[5,3]].y,BrPts.loc[[5,3]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[0,3]].x,BrPts.loc[[0,3]].y,BrPts.loc[[0,3]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[6,7]].x,BrPts.loc[[6,7]].y,BrPts.loc[[6,7]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[7,8]].x,BrPts.loc[[7,8]].y,BrPts.loc[[7,8]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[8,10]].x,BrPts.loc[[8,10]].y,BrPts.loc[[8,10]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[10,11]].x,BrPts.loc[[10,11]].y,BrPts.loc[[10,11]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[9,11]].x,BrPts.loc[[9,11]].y,BrPts.loc[[9,11]].z,linestyle="-",color="k")
ax4.plot(BrPts.loc[[9,6]].x,BrPts.loc[[9,6]].y,BrPts.loc[[9,6]].z,linestyle="-",color="k")

#Lateral Faces
ax4.plot([B1[0],H2[0]],[B1[1],H2[1]],[B1[2],H2[2]],linestyle="-",color="k")
ax4.plot([B0[0],H1[0]],[B0[1],H1[1]],[B0[2],H1[2]],linestyle="-",color="k")
ax4.plot([B4[0],H7[0]],[B4[1],H7[1]],[B4[2],H7[2]],linestyle="-",color="k")
ax4.plot([B5[0],H8[0]],[B5[1],H8[1]],[B5[2],H8[2]],linestyle="-",color="k")
ax4.plot([H7[0],H8[0]],[H7[1],H8[1]],[H7[2],H8[2]],linestyle="-",color="k")
ax4.plot([H2[0],H6[0]],[H2[1],H6[1]],[H2[2],H6[2]],linestyle="-",color="k")
ax4.plot([H1[0],H5[0]],[H1[1],H5[1]],[H1[2],H5[2]],linestyle="-",color="k")
ax4.plot([H7[0],H4[0]],[H7[1],H4[1]],[H7[2],H4[2]],linestyle="-",color="k")
ax4.plot([H8[0],H3[0]],[H8[1],H3[1]],[H8[2],H3[2]],linestyle="-",color="k")
ax4.plot([B_1[0],H6[0]],[B_1[1],H6[1]],[B_1[2],H6[2]],linestyle="-",color="k")
ax4.plot([B_0[0],H5[0]],[B_0[1],H5[1]],[B_0[2],H5[2]],linestyle="-",color="k")
ax4.plot([B_5[0],H3[0]],[B_5[1],H3[1]],[B_5[2],H3[2]],linestyle="-",color="k")
ax4.plot([B_4[0],H4[0]],[B_4[1],H4[1]],[B_4[2],H4[2]],linestyle="-",color="k")
ax4.plot([B2[0],H10[0]],[B2[1],H10[1]],[B2[2],H10[2]],linestyle="-",color="k")
ax4.plot([B3[0],H9[0]],[B3[1],H9[1]],[B3[2],H9[2]],linestyle="-",color="k")
ax4.plot([H10[0],H12[0]],[H10[1],H12[1]],[H10[2],H12[2]],linestyle="-",color="k")
ax4.plot([H9[0],H11[0]],[H9[1],H11[1]],[H9[2],H11[2]],linestyle="-",color="k")
ax4.plot([B_2[0],H12[0]],[B_2[1],H12[1]],[B_2[2],H12[2]],linestyle="-",color="k")
ax4.plot([B_3[0],H11[0]],[B_3[1],H11[1]],[B_3[2],H11[2]],linestyle="-",color="k")
ax4.plot([H6[0],H5[0]],[H6[1],H5[1]],[H6[2],H5[2]],linestyle="-",color="k")
ax4.plot([H3[0],H11[0]],[H3[1],H11[1]],[H3[2],H11[2]],linestyle="-",color="k")
ax4.plot([H4[0],H12[0]],[H4[1],H12[1]],[H4[2],H12[2]],linestyle="-",color="k")
ax4.plot([H9[0],H1[0]],[H9[1],H1[1]],[H9[2],H1[2]],linestyle="-",color="k")
ax4.plot([H10[0],H2[0]],[H10[1],H2[1]],[H10[2],H2[2]],linestyle="-",color="k")


#EDGES OF BRILLOUIN ZONE HEXAGONAL (easier plotting by positions of K points in two heights, see Kh0 definitions)
ax4.plot([1.47,1.47],[0,0],[-0.22,0.22],linestyle="--",color="tab:gray",label="Hexagonal BZ")
ax4.plot([-1.47,-1.47],[0,0],[-0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,1.47],[1.27,0],[0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,1.47],[1.27,0],[-0.22,-0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,0.74],[1.27,1.27],[-0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,-1.47],[1.27,0],[0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,0.74],[1.27,1.27],[0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,-1.47],[1.27,0],[-0.22,-0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,0.74],[1.27,1.27],[-0.22,-0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,-0.74],[1.27,1.27],[-0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,-1.47],[-1.27,0],[0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,-1.47],[-1.27,0],[-0.22,-0.22],linestyle="--",color="tab:gray")
ax4.plot([-0.74,-0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,1.47],[-1.27,0],[0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,-0.74],[-1.27,-1.27],[0.22,0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,1.47],[-1.27,0],[-0.22,-0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,-0.74],[-1.27,-1.27],[-0.22,-0.22],linestyle="--",color="tab:gray")
ax4.plot([0.74,0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="--",color="tab:gray")

#Symmetry points
#Gamma points
#G in rombohedral notation
ax4.scatter(G[0],G[1],G[2],color="k",marker="$G0$",s=160)
ax4.scatter(G1[0],G1[1],G1[2],color="k",marker="$Gr1$",s=160)
ax4.scatter(G2[0],G2[1],G2[2],color="k",marker="$Gr2$",s=160)
ax4.scatter(G3[0],G3[1],G3[2],color="k",marker="$Gr3$",s=160)
ax4.scatter(G4[0],G4[1],G4[2],color="k",marker="$Gr4$",s=160)
ax4.scatter(G5[0],G5[1],G5[2],color="k",marker="$Gr5$",s=160)
ax4.scatter(G6[0],G6[1],G6[2],color="k",marker="$Gr6$",s=160)
ax4.scatter(G_6[0],G_6[1],G_6[2],color="k",marker="$Gr-6$",s=160)
ax4.scatter(G7[0],G7[1],G7[2],color="k",marker="$Gr7$",s=160)
ax4.scatter(G_7[0],G_7[1],G_7[2],color="k",marker="$Gr-7$",s=160)
ax4.scatter(G8[0],G8[1],G8[2],color="k",marker="$Gr8$",s=160)
ax4.scatter(G_8[0],G_8[1],G_8[2],color="k",marker="$Gr-8$",s=160)
ax4.scatter(G9[0],G9[1],G9[2],color="k",marker="$Gr9$",s=160)
#G in hexagonal notation (note differences)
ax4.scatter(Gh1[0],Gh1[1],Gh1[2],color="tab:gray",marker="$Gh1$",s=160)
ax4.scatter(Gh2[0],Gh2[1],Gh2[2],color="tab:gray",marker="$Gh2$",s=160)
ax4.scatter(Gh3[0],Gh3[1],Gh3[2],color="tab:gray",marker="$Gh3$",s=160)
ax4.scatter(Gh4[0],Gh4[1],Gh4[2],color="tab:gray",marker="$Gh4$",s=160)
ax4.scatter(Gh5[0],Gh5[1],Gh5[2],color="tab:gray",marker="$Gh5$",s=160)
ax4.scatter(Gh6[0],Gh6[1],Gh6[2],color="tab:gray",marker="$Gh6$",s=160)
ax4.scatter(Gh7[0],Gh7[1],Gh7[2],color="tab:gray",marker="$Gh7$",s=160)
ax4.scatter(Gh8[0],Gh8[1],Gh8[2],color="tab:gray",marker="$Gh8$",s=160)
ax4.scatter(Gh9[0],Gh9[1],Gh9[2],color="tab:gray",marker="$Gh9$",s=160)
#Z points
#Z points rombohedral notation
ax4.scatter(Z0[0],Z0[1],Z0[2],color="g",marker="$Zr0$",s=160)
ax4.scatter(Z1[0],Z1[1],Z1[2],color="g",marker="$Zr1$",s=160)
ax4.scatter(Z2[0],Z2[1],Z2[2],color="g",marker="$Zr2$",s=160)
ax4.scatter(Z3[0],Z3[1],Z3[2],color="g",marker="$Zr3$",s=160)
ax4.scatter(Z4[0],Z4[1],Z4[2],color="g",marker="$Zr4$",s=160)
ax4.scatter(Z5[0],Z5[1],Z5[2],color="g",marker="$Zr5$",s=160)
ax4.scatter(Z6[0],Z6[1],Z6[2],color="g",marker="$Zr6$",s=160)
ax4.scatter(Z7[0],Z7[1],Z7[2],color="g",marker="$Zr7$",s=160)
#Z points hexagonal notation
ax4.scatter(Zh0[0],Zh0[1],Zh0[2],color="tab:olive",marker="$Zh0$",s=160)
ax4.scatter(Zh1[0],Zh1[1],Zh1[2],color="tab:olive",marker="$Zh1$",s=160)
ax4.scatter(Zh2[0],Zh2[1],Zh2[2],color="tab:olive",marker="$Zh2$",s=160)
ax4.scatter(Zh3[0],Zh3[1],Zh3[2],color="tab:olive",marker="$Zh3$",s=160)
ax4.scatter(Zh4[0],Zh4[1],Zh4[2],color="tab:olive",marker="$Zh4$",s=160)
#L points (center of lateral hexagonal faces)
#L points Rombohedral L notation
ax4.scatter(Lr1[0],Lr1[1],Lr1[2],color="b",marker="$Lr1$",s=160)
ax4.scatter(Lr2[0],Lr2[1],Lr2[2],color="b",marker="$Lr2$",s=160)
ax4.scatter(Lr3[0],Lr3[1],Lr3[2],color="b",marker="$Lr3$",s=160)
ax4.scatter(Lr_1[0],Lr_1[1],Lr_1[2],color="b",marker="$Lr -1$",s=160)
ax4.scatter(Lr_2[0],Lr_2[1],Lr_2[2],color="b",marker="$Lr -2$",s=160)
ax4.scatter(Lr_3[0],Lr_3[1],Lr_3[2],color="b",marker="$Lr -3$",s=160)
#B points (K points in usual notation)
ax4.scatter(B0[0],B0[1],B0[2],color='r',marker="$Br0$",s=160)
ax4.scatter(B1[0],B1[1],B1[2],color='r',marker="$Br1$",s=160)
ax4.scatter(B2[0],B2[1],B2[2],color='r',marker="$Br2$",s=160)
ax4.scatter(B3[0],B3[1],B3[2],color='r',marker="$Br3$",s=160)
ax4.scatter(B4[0],B4[1],B4[2],color='r',marker="$Br4$",s=160)
ax4.scatter(B5[0],B5[1],B5[2],color='r',marker="$Br5$",s=160)
ax4.scatter(B_0[0],B_0[1],B_0[2],color='r',marker="$Br-0$",s=160)
ax4.scatter(B_1[0],B_1[1],B_1[2],color='r',marker="$Br-1$",s=160)
ax4.scatter(B_2[0],B_2[1],B_2[2],color='r',marker="$Br-2$",s=160)
ax4.scatter(B_3[0],B_3[1],B_3[2],color='r',marker="$Br-3$",s=160)
ax4.scatter(B_4[0],B_4[1],B_4[2],color='r',marker="$Br-4$",s=160)
ax4.scatter(B_5[0],B_5[1],B_5[2],color='r',marker="$Br-5$",s=160)
#H points (it is similar to a K in volume)
ax4.scatter(H1[0],H1[1],H1[2],color='tab:orange',marker="$Hr1$",s=160)
ax4.scatter(H2[0],H2[1],H2[2],color='tab:orange',marker="$Hr2$",s=160)
ax4.scatter(H3[0],H3[1],H3[2],color='tab:orange',marker="$Hr3$",s=160)
ax4.scatter(H4[0],H4[1],H4[2],color='tab:orange',marker="$Hr4$",s=160)
ax4.scatter(H5[0],H5[1],H5[2],color='tab:orange',marker="$Hr5$",s=160)
ax4.scatter(H6[0],H6[1],H6[2],color='tab:orange',marker="$Hr6$",s=160)
ax4.scatter(H7[0],H7[1],H7[2],color='tab:orange',marker="$Hr7$",s=160)
ax4.scatter(H8[0],H8[1],H8[2],color='tab:orange',marker="$Hr8$",s=160)
ax4.scatter(H9[0],H9[1],H9[2],color='tab:orange',marker="$Hr9$",s=160)
ax4.scatter(H10[0],H10[1],H10[2],color='tab:orange',marker="$Hr10$",s=160)
ax4.scatter(H11[0],H11[1],H11[2],color='tab:orange',marker="$Hr11$",s=160)
ax4.scatter(H12[0],H12[1],H12[2],color='tab:orange',marker="$Hr12$",s=160)
#M points (P points in other notations)
ax4.scatter(M0[0],M0[1],M0[2],color='c',marker="$Mr0$",s=160)
ax4.scatter(M1[0],M1[1],M1[2],color='c',marker="$Mr1$",s=160)
ax4.scatter(M2[0],M2[1],M2[2],color='c',marker="$Mr2$",s=160)
ax4.scatter(M3[0],M3[1],M3[2],color='c',marker="$Mr3$",s=160)
ax4.scatter(M4[0],M4[1],M4[2],color='c',marker="$Mr4$",s=160)
ax4.scatter(M5[0],M5[1],M5[2],color='c',marker="$Mr5$",s=160)
#F points (centers of rectangular faces)
ax4.scatter(F1[0],F1[1],F1[2],color='g',marker="$Fr1$",s=160)
ax4.scatter(F2[0],F2[1],F2[2],color='g',marker="$Fr2$",s=160)
ax4.scatter(F3[0],F3[1],F3[2],color='g',marker="$Fr3$",s=160)

ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False
ax4.set_xlim(-2.5, 2.5); ax4.set_ylim(-2.5, 2.5); ax4.set_zlim(-2.5, 2.5);
ax4.view_init(90, -90)
#ax4.legend()
plt.show()

##############################################################################
Simplified Figures
##############################################################################

fig5=plt.figure(5)
ax5=plt.subplot()
ax5=plt.axes(projection="3d",proj_type = 'ortho')
#Rhombohedral reciprocal vectors
ax5.plot(br.loc[[0,1]].x,br.loc[[0,1]].y,br.loc[[0,1]].z,label="Rhombohedral Reciprocal Axes",linestyle=":",color="r")
ax5.plot(br.loc[[0,2]].x,br.loc[[0,2]].y,br.loc[[0,2]].z,linestyle=":",color="r")
ax5.plot(br.loc[[0,3]].x,br.loc[[0,3]].y,br.loc[[0,3]].z,linestyle=":",color="r")
#Hexagonal reciprocal vectors
ax5.plot(bh.loc[[0,1]].x,bh.loc[[0,1]].y,bh.loc[[0,1]].z,label="hexagonal Reciprocal Axes",linestyle=":",color="k")
ax5.plot(bh.loc[[0,2]].x,bh.loc[[0,2]].y,bh.loc[[0,2]].z,linestyle=":",color="k")
ax5.plot(bh.loc[[0,3]].x,bh.loc[[0,3]].y,bh.loc[[0,3]].z,linestyle=":",color="k")
#directino 111 in Rhombohedral axes
ax5.plot([0,Db111[0]],[0,Db111[1]],[0,Db111[2]],linestyle="--",label="[111] reciprocal",color="y")

#Symmetry points
#Gamma points
#G in rombohedral notation
ax5.scatter(G[0],G[1],G[2],color="k",marker="o",s=160,label="Gamma Points")
ax5.scatter(G1[0],G1[1],G1[2],color="k",marker="o",s=60,alpha=0.5)
# ax5.scatter(G2[0],G2[1],G2[2],color="k",marker="o",s=60)
# ax5.scatter(G3[0],G3[1],G3[2],color="k",marker="o",s=60)
ax5.scatter(G4[0],G4[1],G4[2],color="k",marker="o",s=60,alpha=0.5)
ax5.scatter(G5[0],G5[1],G5[2],color="k",marker="o",s=60,alpha=0.5)
# ax5.scatter(G6[0],G6[1],G6[2],color="k",marker="o",s=60)
ax5.scatter(G_6[0],G_6[1],G_6[2],color="k",marker="o",s=60,alpha=0.5)
# ax5.scatter(G7[0],G7[1],G7[2],color="k",marker="o",s=60)
# ax5.scatter(G_7[0],G_7[1],G_7[2],color="k",marker="o",s=60)
# ax5.scatter(G8[0],G8[1],G8[2],color="k",marker="o",s=60)
# ax5.scatter(G_8[0],G_8[1],G_8[2],color="k",marker="o",s=60)
ax5.scatter(G9[0],G9[1],G9[2],color="k",marker="o",s=60,alpha=0.5)
#G in hexagonal notation
ax5.scatter(Gh1[0],Gh1[1],Gh1[2],color="tab:gray",marker=".",s=50)
ax5.scatter(Gh2[0],Gh2[1],Gh2[2],color="tab:gray",marker=".",s=50,alpha=0.5)
ax5.scatter(Gh3[0],Gh3[1],Gh3[2],color="tab:gray",marker=".",s=50,alpha=0.5)
# ax5.scatter(Gh4[0],Gh4[1],Gh4[2],color="tab:gray",marker=".",s=50)
# ax5.scatter(Gh5[0],Gh5[1],Gh5[2],color="tab:gray",marker=".",s=50)
# ax5.scatter(Gh6[0],Gh6[1],Gh6[2],color="tab:gray",marker=".",s=50)
# ax5.scatter(Gh7[0],Gh7[1],Gh7[2],color="tab:gray",marker=".",s=50)
# ax5.scatter(Gh8[0],Gh8[1],Gh8[2],color="tab:gray",marker=".",s=50)
ax5.scatter(Gh9[0],Gh9[1],Gh9[2],color="tab:gray",marker=".",s=50,alpha=0.5)

#Z points
#Z points rombohedral notation
ax5.scatter(Z0[0],Z0[1],Z0[2],color="green",marker="^",s=160,label="Z (T) Points")
ax5.scatter(Z1[0],Z1[1],Z1[2],color="green",marker="^",s=60,alpha=0.5)
ax5.scatter(Z2[0],Z2[1],Z2[2],color="green",marker="^",s=60,alpha=0.5)
ax5.scatter(Z3[0],Z3[1],Z3[2],color="green",marker="^",s=60,alpha=0.5)
ax5.scatter(Z4[0],Z4[1],Z4[2],color="green",marker="^",s=60,alpha=0.5)
ax5.scatter(Z5[0],Z5[1],Z5[2],color="green",marker="^",s=60,alpha=0.5)
# ax5.scatter(Z6[0],Z6[1],Z6[2],color="g",marker="^",s=60)
# ax5.scatter(Z7[0],Z7[1],Z7[2],color="g",marker="^",s=60)
#Z points hexagonal notation
ax5.scatter(Zh0[0],Zh0[1],Zh0[2],color="lightgreen",marker="2",s=50)
ax5.scatter(Zh1[0],Zh1[1],Zh1[2],color="lightgreen",marker="2",s=50,alpha=0.5)
ax5.scatter(Zh2[0],Zh2[1],Zh2[2],color="lightgreen",marker="2",s=50,alpha=0.5)
ax5.scatter(Zh3[0],Zh3[1],Zh3[2],color="lightgreen",marker="2",s=50,alpha=0.5)
ax5.scatter(Zh4[0],Zh4[1],Zh4[2],color="lightgreen",marker="2",s=50,alpha=0.5)


#L points (center of lateral hexagonal faces)
#L points Rombohedral L notation
ax5.scatter(Lr1[0],Lr1[1],Lr1[2],color="lightblue",marker="h",s=60,label="L Points",alpha=0.5)
ax5.scatter(Lr2[0],Lr2[1],Lr2[2],color="lightblue",marker="h",s=60,alpha=0.5)
ax5.scatter(Lr3[0],Lr3[1],Lr3[2],color="lightblue",marker="h",s=60,alpha=0.5)
ax5.scatter(Lr_1[0],Lr_1[1],Lr_1[2],color="lightblue",marker="h",s=60,alpha=0.5)
ax5.scatter(Lr_2[0],Lr_2[1],Lr_2[2],color="lightblue",marker="h",s=60,alpha=0.5)
ax5.scatter(Lr_3[0],Lr_3[1],Lr_3[2],color="lightblue",marker="h",s=60,alpha=0.5)

#B points (K points in usual notation)
ax5.scatter(B0[0],B0[1],B0[2],color='r',marker="$B$",s=150)
ax5.scatter(B1[0],B1[1],B1[2],color='r',marker="$B$",s=150)
ax5.scatter(B2[0],B2[1],B2[2],color='r',marker=".",s=60,alpha=0.5)
ax5.scatter(B3[0],B3[1],B3[2],color='r',marker=".",s=60,alpha=0.5)
ax5.scatter(B4[0],B4[1],B4[2],color='r',marker=".",s=60,alpha=0.5)
ax5.scatter(B5[0],B5[1],B5[2],color='r',marker=".",s=60,alpha=0.5)

ax5.scatter(B_0[0],B_0[1],B_0[2],color='coral',marker=".",s=60,alpha=0.5)
ax5.scatter(B_1[0],B_1[1],B_1[2],color='coral',marker=".",s=60,alpha=0.5)
ax5.scatter(B_2[0],B_2[1],B_2[2],color='coral',marker=".",s=60,alpha=0.5)
ax5.scatter(B_3[0],B_3[1],B_3[2],color='coral',marker=".",s=60,alpha=0.5)
ax5.scatter(B_4[0],B_4[1],B_4[2],color='coral',marker=".",s=60,alpha=0.5)
ax5.scatter(B_5[0],B_5[1],B_5[2],color='coral',marker=".",s=60,alpha=0.5)


#K points hexagonal notation
ax5.scatter(Kh0[0],Kh0[1],Kh0[2],color='r',marker="$K$",s=150)



#H points (it is similar to a K in volume)
ax5.scatter(H1[0],H1[1],H1[2],color='tab:orange',marker=".",s=50,label="H (B) Points")
ax5.scatter(H2[0],H2[1],H2[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H3[0],H3[1],H3[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H4[0],H4[1],H4[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H5[0],H5[1],H5[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H6[0],H6[1],H6[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H7[0],H7[1],H7[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H8[0],H8[1],H8[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H9[0],H9[1],H9[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H10[0],H10[1],H10[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H11[0],H11[1],H11[2],color='tab:orange',marker=".",s=50)
ax5.scatter(H12[0],H12[1],H12[2],color='tab:orange',marker=".",s=50)

#M points (P points in other notations)
ax5.scatter(M0[0],M0[1],M0[2],color='b',marker="$M$",s=150)
ax5.scatter(M1[0],M1[1],M1[2],color='b',marker=".",s=60,alpha=0.5)
ax5.scatter(M2[0],M2[1],M2[2],color='b',marker=".",s=60,alpha=0.5)
ax5.scatter(M3[0],M3[1],M3[2],color='b',marker=".",s=60,alpha=0.5)
ax5.scatter(M4[0],M4[1],M4[2],color='b',marker="$M'$",s=150)
ax5.scatter(M5[0],M5[1],M5[2],color='b',marker=".",s=60,alpha=0.5)

ax5.scatter(Mh0[0],Mh0[1],Mh0[2],color='indigo',marker="$M$",s=150)
ax5.scatter(Mph0[0],Mph0[1],Mph0[2],color='indigo',marker="$M'$",s=150)


#F points (centers of rectangular faces)
ax5.scatter(F1[0],F1[1],F1[2],color='lavender',marker="s",s=50,label="F Points",alpha=0.5)
ax5.scatter(F2[0],F2[1],F2[2],color='lavender',marker="s",s=50,alpha=0.5)
ax5.scatter(F3[0],F3[1],F3[2],color='lavender',marker="s",s=50,alpha=0.5)

#111 Face of Rhombohedral Brillouin Zone
ax5.plot(BrPts.loc[[0,1]].x,BrPts.loc[[0,1]].y,BrPts.loc[[0,1]].z,linestyle="-",color="k",label="Rhombohedral BZ")
ax5.plot(BrPts.loc[[2,1]].x,BrPts.loc[[2,1]].y,BrPts.loc[[2,1]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[2,4]].x,BrPts.loc[[2,4]].y,BrPts.loc[[2,4]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[5,4]].x,BrPts.loc[[5,4]].y,BrPts.loc[[5,4]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[5,3]].x,BrPts.loc[[5,3]].y,BrPts.loc[[5,3]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[0,3]].x,BrPts.loc[[0,3]].y,BrPts.loc[[0,3]].z,linestyle="-",color="k")

ax5.plot(BrPts.loc[[6,7]].x,BrPts.loc[[6,7]].y,BrPts.loc[[6,7]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[7,8]].x,BrPts.loc[[7,8]].y,BrPts.loc[[7,8]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[8,10]].x,BrPts.loc[[8,10]].y,BrPts.loc[[8,10]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[10,11]].x,BrPts.loc[[10,11]].y,BrPts.loc[[10,11]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[9,11]].x,BrPts.loc[[9,11]].y,BrPts.loc[[9,11]].z,linestyle="-",color="k")
ax5.plot(BrPts.loc[[9,6]].x,BrPts.loc[[9,6]].y,BrPts.loc[[9,6]].z,linestyle="-",color="k")
#Lateral Faces
ax5.plot([B1[0],H2[0]],[B1[1],H2[1]],[B1[2],H2[2]],linestyle="-",color="k")
ax5.plot([B0[0],H1[0]],[B0[1],H1[1]],[B0[2],H1[2]],linestyle="-",color="k")
ax5.plot([B4[0],H7[0]],[B4[1],H7[1]],[B4[2],H7[2]],linestyle="-",color="k")
ax5.plot([B5[0],H8[0]],[B5[1],H8[1]],[B5[2],H8[2]],linestyle="-",color="k")
ax5.plot([H7[0],H8[0]],[H7[1],H8[1]],[H7[2],H8[2]],linestyle="-",color="k")
ax5.plot([H2[0],H6[0]],[H2[1],H6[1]],[H2[2],H6[2]],linestyle="-",color="k")
ax5.plot([H1[0],H5[0]],[H1[1],H5[1]],[H1[2],H5[2]],linestyle="-",color="k")
ax5.plot([H7[0],H4[0]],[H7[1],H4[1]],[H7[2],H4[2]],linestyle="-",color="k")
ax5.plot([H8[0],H3[0]],[H8[1],H3[1]],[H8[2],H3[2]],linestyle="-",color="k")
ax5.plot([B_1[0],H6[0]],[B_1[1],H6[1]],[B_1[2],H6[2]],linestyle="-",color="k")
ax5.plot([B_0[0],H5[0]],[B_0[1],H5[1]],[B_0[2],H5[2]],linestyle="-",color="k")
ax5.plot([B_5[0],H3[0]],[B_5[1],H3[1]],[B_5[2],H3[2]],linestyle="-",color="k")
ax5.plot([B_4[0],H4[0]],[B_4[1],H4[1]],[B_4[2],H4[2]],linestyle="-",color="k")
ax5.plot([B2[0],H10[0]],[B2[1],H10[1]],[B2[2],H10[2]],linestyle="-",color="k")
ax5.plot([B3[0],H9[0]],[B3[1],H9[1]],[B3[2],H9[2]],linestyle="-",color="k")
ax5.plot([H10[0],H12[0]],[H10[1],H12[1]],[H10[2],H12[2]],linestyle="-",color="k")
ax5.plot([H9[0],H11[0]],[H9[1],H11[1]],[H9[2],H11[2]],linestyle="-",color="k")
ax5.plot([B_2[0],H12[0]],[B_2[1],H12[1]],[B_2[2],H12[2]],linestyle="-",color="k")
ax5.plot([B_3[0],H11[0]],[B_3[1],H11[1]],[B_3[2],H11[2]],linestyle="-",color="k")
ax5.plot([H6[0],H5[0]],[H6[1],H5[1]],[H6[2],H5[2]],linestyle="-",color="k")
ax5.plot([H3[0],H11[0]],[H3[1],H11[1]],[H3[2],H11[2]],linestyle="-",color="k")
ax5.plot([H4[0],H12[0]],[H4[1],H12[1]],[H4[2],H12[2]],linestyle="-",color="k")
ax5.plot([H9[0],H1[0]],[H9[1],H1[1]],[H9[2],H1[2]],linestyle="-",color="k")
ax5.plot([H10[0],H2[0]],[H10[1],H2[1]],[H10[2],H2[2]],linestyle="-",color="k")


#EDGES OF BRILLOUIN ZONE HEXAGONAL
ax5.plot([1.47,1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray",label="Hexagonal BZ")
ax5.plot([-1.47,-1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,-1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,0.74],[1.27,1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,-1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,0.74],[1.27,1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,-0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,-1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,-1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax5.plot([-0.74,-0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,-0.74],[-1.27,-1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,-0.74],[-1.27,-1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax5.plot([0.74,0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")

#PATH CALCULATED
#Gamma-T(Z)-H2/H0-L-Gamma-S0/S2-F-Gamma
ax5.plot([G[0],Z0[0]],[G[1],Z0[1]],[G[2],Z0[2]],linewidth=5,color="coral",alpha=0.4)#G-T(Z)
# ax5.plot([B0[0],Z0[0]],[B0[1],Z0[1]],[B0[2],Z0[2]],linewidth=10,color="coral",alpha=0.4)#T(Z)-H2(B)
ax5.plot([Z0[0],H1[0]],[Z0[1],H1[1]],[Z0[2],H1[2]],linewidth=5,color="coral",alpha=0.4)#Z- H2-H0
ax5.plot([H1[0],Lr1[0]],[H1[1],Lr1[1]],[H1[2],Lr1[2]],linewidth=5,color="coral",alpha=0.4)#H0-L
ax5.plot([G[0],Lr1[0]],[G[1],Lr1[1]],[G[2],Lr1[2]],linewidth=5,color="coral",alpha=0.4)#L-G
ax5.plot([G[0],B0[0]],[G[1],B0[1]],[G[2],0],linewidth=5,color="coral",alpha=0.4)#G-S0
# ax5.plot([B0[0],B0[0]],[B0[1],B0[1]],[0.33,0],linewidth=10,color="coral",alpha=0.4)#S0-S2
ax5.plot([B0[0],F1[0]],[B0[1],F1[1]],[0,F1[2]],linewidth=5,color="coral",alpha=0.4)#S0-F
ax5.plot([G[0],F1[0]],[G[1],F1[1]],[G[2],F1[2]],linewidth=5,color="coral",alpha=0.4)#F-G


ax5.xaxis.pane.fill = False
ax5.yaxis.pane.fill = False
ax5.zaxis.pane.fill = False
ax5.set_xlim(-3, 3); ax5.set_ylim(-3, 3); ax5.set_zlim(-3, 3);
ax5.view_init(90, -90)
ax5.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax5.grid(False)
#ax5.legend()
plt.show()

##############################################################################################
#Theoretical Paths:
##############################################################################################

#Definition of points in the correct notation for the plotting of Paths in the BZ
Ght=0*bh1+0*bh2+0*bh3
Mht=0.5*bh1
MG=np.linspace(Mht,Ght,num=41)
Aht=0.5*bh3
Kht=0.33*bh1+0.33*bh2
Lht=0.5*bh1+0.5*bh3
Hht=0.33*bh1+0.33*bh2+0.5*bh3


fig6=plt.figure(6) #Hezagonal cells
ax6=plt.subplot()
ax6=plt.axes(projection="3d",proj_type = 'ortho')

#EDGES OF BRILLOUIN ZONE HEXAGONAL (Degfined with K points at the faces see Kh0)
ax6.plot([1.47,1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray",label="Hexagonal BZ")
ax6.plot([-1.47,-1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,-1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,0.74],[1.27,1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,-1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,0.74],[1.27,1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,-0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,-1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,-1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax6.plot([-0.74,-0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,-0.74],[-1.27,-1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,-0.74],[-1.27,-1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax6.plot([0.74,0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")

#Rhombohedral Brillouin Zone
ax6.plot(BrPts.loc[[0,1]].x,BrPts.loc[[0,1]].y,BrPts.loc[[0,1]].z,linestyle="-",color="k",label="Rhombohedral BZ")
ax6.plot(BrPts.loc[[2,1]].x,BrPts.loc[[2,1]].y,BrPts.loc[[2,1]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[2,4]].x,BrPts.loc[[2,4]].y,BrPts.loc[[2,4]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[5,4]].x,BrPts.loc[[5,4]].y,BrPts.loc[[5,4]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[5,3]].x,BrPts.loc[[5,3]].y,BrPts.loc[[5,3]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[0,3]].x,BrPts.loc[[0,3]].y,BrPts.loc[[0,3]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[6,7]].x,BrPts.loc[[6,7]].y,BrPts.loc[[6,7]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[7,8]].x,BrPts.loc[[7,8]].y,BrPts.loc[[7,8]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[8,10]].x,BrPts.loc[[8,10]].y,BrPts.loc[[8,10]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[10,11]].x,BrPts.loc[[10,11]].y,BrPts.loc[[10,11]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[9,11]].x,BrPts.loc[[9,11]].y,BrPts.loc[[9,11]].z,linestyle="-",color="k")
ax6.plot(BrPts.loc[[9,6]].x,BrPts.loc[[9,6]].y,BrPts.loc[[9,6]].z,linestyle="-",color="k")
ax6.plot([B1[0],H2[0]],[B1[1],H2[1]],[B1[2],H2[2]],linestyle="-",color="k")
ax6.plot([B0[0],H1[0]],[B0[1],H1[1]],[B0[2],H1[2]],linestyle="-",color="k")
ax6.plot([B4[0],H7[0]],[B4[1],H7[1]],[B4[2],H7[2]],linestyle="-",color="k")
ax6.plot([B5[0],H8[0]],[B5[1],H8[1]],[B5[2],H8[2]],linestyle="-",color="k")
ax6.plot([H7[0],H8[0]],[H7[1],H8[1]],[H7[2],H8[2]],linestyle="-",color="k")
ax6.plot([H2[0],H6[0]],[H2[1],H6[1]],[H2[2],H6[2]],linestyle="-",color="k")
ax6.plot([H1[0],H5[0]],[H1[1],H5[1]],[H1[2],H5[2]],linestyle="-",color="k")
ax6.plot([H7[0],H4[0]],[H7[1],H4[1]],[H7[2],H4[2]],linestyle="-",color="k")
ax6.plot([H8[0],H3[0]],[H8[1],H3[1]],[H8[2],H3[2]],linestyle="-",color="k")
ax6.plot([B_1[0],H6[0]],[B_1[1],H6[1]],[B_1[2],H6[2]],linestyle="-",color="k")
ax6.plot([B_0[0],H5[0]],[B_0[1],H5[1]],[B_0[2],H5[2]],linestyle="-",color="k")
ax6.plot([B_5[0],H3[0]],[B_5[1],H3[1]],[B_5[2],H3[2]],linestyle="-",color="k")
ax6.plot([B_4[0],H4[0]],[B_4[1],H4[1]],[B_4[2],H4[2]],linestyle="-",color="k")
ax6.plot([B2[0],H10[0]],[B2[1],H10[1]],[B2[2],H10[2]],linestyle="-",color="k")
ax6.plot([B3[0],H9[0]],[B3[1],H9[1]],[B3[2],H9[2]],linestyle="-",color="k")
ax6.plot([H10[0],H12[0]],[H10[1],H12[1]],[H10[2],H12[2]],linestyle="-",color="k")
ax6.plot([H9[0],H11[0]],[H9[1],H11[1]],[H9[2],H11[2]],linestyle="-",color="k")
ax6.plot([B_2[0],H12[0]],[B_2[1],H12[1]],[B_2[2],H12[2]],linestyle="-",color="k")
ax6.plot([B_3[0],H11[0]],[B_3[1],H11[1]],[B_3[2],H11[2]],linestyle="-",color="k")
ax6.plot([H6[0],H5[0]],[H6[1],H5[1]],[H6[2],H5[2]],linestyle="-",color="k")
ax6.plot([H3[0],H11[0]],[H3[1],H11[1]],[H3[2],H11[2]],linestyle="-",color="k")
ax6.plot([H4[0],H12[0]],[H4[1],H12[1]],[H4[2],H12[2]],linestyle="-",color="k")
ax6.plot([H9[0],H1[0]],[H9[1],H1[1]],[H9[2],H1[2]],linestyle="-",color="k")
ax6.plot([H10[0],H2[0]],[H10[1],H2[1]],[H10[2],H2[2]],linestyle="-",color="k")

#MEASURED Path examples:
#hv150 is kz Gamma (+/)-0.22 (A)
ax6.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[-0.22,-0.22],linewidth=5,color="red",alpha=0.3) #GM 150eV
ax6.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.22,0.22],linewidth=5,color="red",alpha=0.3)
ax6.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[-0.22,-0.22],linewidth=5,color="red",alpha=0.3)#GK 150eV
ax6.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.22,0.22],linewidth=5,color="red",alpha=0.3)
#hv163 is kz Gamma 0 (G)
ax6.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0,0],linewidth=5,color="red",alpha=0.3) #GM 163eV
#hv132 is kz Z (T) 0.66 (Z)
ax6.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.66,0.66],linewidth=5,color="red",alpha=0.3) #GM 132eV

#Lambrecht hexagonal path
ax6.plot([0.5*bh.loc[1].x,0],[0.5*bh.loc[1].y,0],[0.5*bh.loc[1].z,0],linewidth=5,color="blue",alpha=0.4)#M-G
ax6.plot([0,0.33*bh.loc[1].x+0.33*bh.loc[2].x],[0,0.33*bh.loc[1].y+0.33*bh.loc[2].y],[0,0.33*bh.loc[1].z+0.33*bh.loc[2].z],linewidth=5,color="blue",alpha=0.4)#G-K
ax6.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x,0.5*bh.loc[1].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y,0.5*bh.loc[1].y],[0.33*bh.loc[1].z+0.33*bh.loc[2].z,0.5*bh.loc[1].z],linewidth=5,color="blue",alpha=0.4)#K-M
ax6.plot([0.5*bh.loc[1].x,0.5*bh.loc[1].x+0.5*bh.loc[3].x],[0.5*bh.loc[1].y,0.5*bh.loc[1].y+0.5*bh.loc[3].y],[0,0.22],linewidth=5,color="blue",alpha=0.4)#M-L
ax6.plot([0.5*bh.loc[1].x+0.5*bh.loc[3].x,0.5*bh.loc[3].x],[0.5*bh.loc[1].y+0.5*bh.loc[3].y,0.5*bh.loc[3].y],[0.22,0.22],linewidth=5,color="blue",alpha=0.4)#L-Z
ax6.plot([0.5*bh.loc[3].x,0],[0.5*bh.loc[3].y,0],[0.22,0],linewidth=5,color="blue",alpha=0.4)#Z-G
ax6.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x,0.33*bh.loc[1].x+0.33*bh.loc[2].x+0.5*bh.loc[3].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y,0.33*bh.loc[1].y+0.33*bh.loc[2].y+0.5*bh.loc[3].y],[0,0.22],linewidth=5,color="blue",alpha=0.4)#K-B
ax6.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x+0.5*bh.loc[3].x,0.5*bh.loc[3].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y+0.5*bh.loc[3].y,0.5*bh.loc[3].y],[0.22,0.22],linewidth=5,color="blue",alpha=0.4)#B-Z

#Calculated TOP VB
# VBmaxTheory=np.array([0.456,0,0])
# ax8.scatter(VBmaxTheory[0],VBmaxTheory[1],VBmaxTheory[2],color="red",marker="*",s=100)

ax6.xaxis.pane.fill = False
ax6.yaxis.pane.fill = False
ax6.zaxis.pane.fill = False
ax6.set_xlim(-2, 2); ax6.set_ylim(-2, 2); ax6.set_zlim(-2, 2);
ax6.view_init(45, -45)
ax6.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax6.grid(False)
#ax6.legend()
plt.show()


########### Paths in all notations ###############


fig7=plt.figure(7)
ax7=plt.subplot()
ax7=plt.axes(projection="3d",proj_type = 'ortho')

#EDGES OF BRILLOUIN ZONE HEXAGONAL
ax7.plot([1.47,1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray",label="Hexagonal BZ")
ax7.plot([-1.47,-1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,-1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,0.74],[1.27,1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,-1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,0.74],[1.27,1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,-0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,-1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,-1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax7.plot([-0.74,-0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,-0.74],[-1.27,-1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,-0.74],[-1.27,-1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax7.plot([0.74,0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")

#Rhombohedral Brillouin Zone
ax7.plot(BrPts.loc[[0,1]].x,BrPts.loc[[0,1]].y,BrPts.loc[[0,1]].z,linestyle="-",color="k",label="Rhombohedral BZ")
ax7.plot(BrPts.loc[[2,1]].x,BrPts.loc[[2,1]].y,BrPts.loc[[2,1]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[2,4]].x,BrPts.loc[[2,4]].y,BrPts.loc[[2,4]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[5,4]].x,BrPts.loc[[5,4]].y,BrPts.loc[[5,4]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[5,3]].x,BrPts.loc[[5,3]].y,BrPts.loc[[5,3]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[0,3]].x,BrPts.loc[[0,3]].y,BrPts.loc[[0,3]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[6,7]].x,BrPts.loc[[6,7]].y,BrPts.loc[[6,7]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[7,8]].x,BrPts.loc[[7,8]].y,BrPts.loc[[7,8]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[8,10]].x,BrPts.loc[[8,10]].y,BrPts.loc[[8,10]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[10,11]].x,BrPts.loc[[10,11]].y,BrPts.loc[[10,11]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[9,11]].x,BrPts.loc[[9,11]].y,BrPts.loc[[9,11]].z,linestyle="-",color="k")
ax7.plot(BrPts.loc[[9,6]].x,BrPts.loc[[9,6]].y,BrPts.loc[[9,6]].z,linestyle="-",color="k")
ax7.plot([B1[0],H2[0]],[B1[1],H2[1]],[B1[2],H2[2]],linestyle="-",color="k")
ax7.plot([B0[0],H1[0]],[B0[1],H1[1]],[B0[2],H1[2]],linestyle="-",color="k")
ax7.plot([B4[0],H7[0]],[B4[1],H7[1]],[B4[2],H7[2]],linestyle="-",color="k")
ax7.plot([B5[0],H8[0]],[B5[1],H8[1]],[B5[2],H8[2]],linestyle="-",color="k")
ax7.plot([H7[0],H8[0]],[H7[1],H8[1]],[H7[2],H8[2]],linestyle="-",color="k")
ax7.plot([H2[0],H6[0]],[H2[1],H6[1]],[H2[2],H6[2]],linestyle="-",color="k")
ax7.plot([H1[0],H5[0]],[H1[1],H5[1]],[H1[2],H5[2]],linestyle="-",color="k")
ax7.plot([H7[0],H4[0]],[H7[1],H4[1]],[H7[2],H4[2]],linestyle="-",color="k")
ax7.plot([H8[0],H3[0]],[H8[1],H3[1]],[H8[2],H3[2]],linestyle="-",color="k")
ax7.plot([B_1[0],H6[0]],[B_1[1],H6[1]],[B_1[2],H6[2]],linestyle="-",color="k")
ax7.plot([B_0[0],H5[0]],[B_0[1],H5[1]],[B_0[2],H5[2]],linestyle="-",color="k")
ax7.plot([B_5[0],H3[0]],[B_5[1],H3[1]],[B_5[2],H3[2]],linestyle="-",color="k")
ax7.plot([B_4[0],H4[0]],[B_4[1],H4[1]],[B_4[2],H4[2]],linestyle="-",color="k")
ax7.plot([B2[0],H10[0]],[B2[1],H10[1]],[B2[2],H10[2]],linestyle="-",color="k")
ax7.plot([B3[0],H9[0]],[B3[1],H9[1]],[B3[2],H9[2]],linestyle="-",color="k")
ax7.plot([H10[0],H12[0]],[H10[1],H12[1]],[H10[2],H12[2]],linestyle="-",color="k")
ax7.plot([H9[0],H11[0]],[H9[1],H11[1]],[H9[2],H11[2]],linestyle="-",color="k")
ax7.plot([B_2[0],H12[0]],[B_2[1],H12[1]],[B_2[2],H12[2]],linestyle="-",color="k")
ax7.plot([B_3[0],H11[0]],[B_3[1],H11[1]],[B_3[2],H11[2]],linestyle="-",color="k")
ax7.plot([H6[0],H5[0]],[H6[1],H5[1]],[H6[2],H5[2]],linestyle="-",color="k")
ax7.plot([H3[0],H11[0]],[H3[1],H11[1]],[H3[2],H11[2]],linestyle="-",color="k")
ax7.plot([H4[0],H12[0]],[H4[1],H12[1]],[H4[2],H12[2]],linestyle="-",color="k")
ax7.plot([H9[0],H1[0]],[H9[1],H1[1]],[H9[2],H1[2]],linestyle="-",color="k")
ax7.plot([H10[0],H2[0]],[H10[1],H2[1]],[H10[2],H2[2]],linestyle="-",color="k")

#MEASURED
#hv150 is kz Gamma (+/)-0.22 (A)
ax7.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[-0.22,-0.22],linewidth=5,color="red",alpha=0.3) #GM 150eV
ax7.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.22,0.22],linewidth=5,color="red",alpha=0.3)
ax7.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[-0.22,-0.22],linewidth=5,color="red",alpha=0.3)#GK 150eV
ax7.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.22,0.22],linewidth=5,color="red",alpha=0.3)
#hv163 is kz Gamma 0 (G)
ax7.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0,0],linewidth=5,color="red",alpha=0.3) #GM 163eV
ax7.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0,0],linewidth=5,color="red",alpha=0.3)#GK 150eV
#hv132 is kz Z (T) 0.66 (Z)
ax7.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.66,0.66],linewidth=5,color="red",alpha=0.3) #GM 132eV
ax7.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.66,0.66],linewidth=5,color="red",alpha=0.3)#GK 132eV

ax7.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.33,0.33],linewidth=5,color="red",alpha=0.3) #GM 
ax7.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.33,0.33],linewidth=5,color="red",alpha=0.3)#GK 

#Measured TOP VB
#150eV LCO6 FS0
VBmax150=bh1*0.215+bh3*0.6
ax7.scatter(VBmax150[0],VBmax150[1],VBmax150[2], color="red",marker="*",s=100)

#Lambrecht hexagonal path
ax7.plot([0.5*bh.loc[1].x,0],[0.5*bh.loc[1].y,0],[0.5*bh.loc[1].z,0],linewidth=5,color="blue",alpha=0.4)#M-G
ax7.plot([0,0.33*bh.loc[1].x+0.33*bh.loc[2].x],[0,0.33*bh.loc[1].y+0.33*bh.loc[2].y],[0,0.33*bh.loc[1].z+0.33*bh.loc[2].z],linewidth=5,color="blue",alpha=0.4)#G-K
ax7.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x,0.5*bh.loc[1].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y,0.5*bh.loc[1].y],[0.33*bh.loc[1].z+0.33*bh.loc[2].z,0.5*bh.loc[1].z],linewidth=5,color="blue",alpha=0.4)#K-M
ax7.plot([0.5*bh.loc[1].x,0.5*bh.loc[1].x+0.5*bh.loc[3].x],[0.5*bh.loc[1].y,0.5*bh.loc[1].y+0.5*bh.loc[3].y],[0,0.22],linewidth=5,color="blue",alpha=0.4)#M-L
ax7.plot([0.5*bh.loc[1].x+0.5*bh.loc[3].x,0.5*bh.loc[3].x],[0.5*bh.loc[1].y+0.5*bh.loc[3].y,0.5*bh.loc[3].y],[0.22,0.22],linewidth=5,color="blue",alpha=0.4)#L-Z
ax7.plot([0.5*bh.loc[3].x,0],[0.5*bh.loc[3].y,0],[0.22,0],linewidth=5,color="blue",alpha=0.4)#Z-G
#ax7.plot([0.5*bh.loc[2].x,0],[0.5*bh.loc[2].y,0],[0.5*bh.loc[2].z,0],linewidth=5,color="blue",alpha=0.4)#M'-G
ax7.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x,0.33*bh.loc[1].x+0.33*bh.loc[2].x+0.5*bh.loc[3].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y,0.33*bh.loc[1].y+0.33*bh.loc[2].y+0.5*bh.loc[3].y],[0,0.22],linewidth=5,color="blue",alpha=0.4)#K-B
ax7.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x+0.5*bh.loc[3].x,0.5*bh.loc[3].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y+0.5*bh.loc[3].y,0.5*bh.loc[3].y],[0.22,0.22],linewidth=5,color="blue",alpha=0.4)#B-Z

#Lambrecht rhombohedral path
ax7.plot([0.5*br.loc[1].x+0.5*br.loc[2].x+0.5*br.loc[3].x,0],[0.5*br.loc[1].y+0.5*br.loc[2].y+0.5*br.loc[3].y,0],[0.5*br.loc[1].z+0.5*br.loc[2].z+0.5*br.loc[3].z,0],linewidth=5,color="green",alpha=0.4)#G-T
ax7.plot([T_t[0], 0],[T_t[1], 0],[T_t[2], 0],linewidth=5,color="green",alpha=0.4)#G-T
ax7.plot([T_t[0], H0_t[0]],[T_t[1], H0_t[1]],[T_t[2], H0_t[2]],linewidth=5,color="green",alpha=0.4)#T-H
ax7.plot([Lht[0], H0_t[0]],[Lht[1], H0_t[1]],[Lht[2], H0_t[2]],linewidth=5,color="green",alpha=0.4)#H-L
ax7.plot([Lht[0], 0],[Lht[1], 0],[Lht[2], 0],linewidth=5,color="green",alpha=0.4)#L-G
ax7.plot([S0_t[0], 0],[S0_t[1], 0],[S0_t[2], 0],linewidth=5,color="green",alpha=0.4)#G-S
ax7.plot([S0_t[0], F1[0]],[S0_t[1], F1[1]],[S0_t[2], F1[2]],linewidth=5,color="green",alpha=0.4)#S-F
ax7.plot([0, F1[0]],[0, F1[1]],[0, F1[2]],linewidth=5,color="green",alpha=0.4)#F-G
#Calculated TOP VB
VBmaxTheory=np.array([0.456,0,0.56])
ax7.scatter(VBmaxTheory[0],VBmaxTheory[1],VBmaxTheory[2],color="green",marker="*",s=100)

ax7.xaxis.pane.fill = False
ax7.yaxis.pane.fill = False
ax7.zaxis.pane.fill = False
ax7.set_xlim(-2, 2); ax7.set_ylim(-2, 2); ax7.set_zlim(-2, 2);
ax7.view_init(45, -45)
ax7.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax7.grid(False)
#ax7.legend()
plt.show()


############################################################################
#Additional figure with paths and symmetry points, choose the representation wanted and comment the other.

fig10=plt.figure(10)
ax10=plt.subplot()
ax10=plt.axes(projection="3d",proj_type = 'ortho')


#EDGES OF BRILLOUIN ZONE HEXAGONAL
ax10.plot([1.47,1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray",label="Hexagonal BZ")
ax10.plot([-1.47,-1.47],[0,0],[-0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,-1.47],[1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,0.74],[1.27,1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,-1.47],[1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,0.74],[1.27,1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,-0.74],[1.27,1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,-1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,-1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax10.plot([-0.74,-0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,1.47],[-1.27,0],[0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,-0.74],[-1.27,-1.27],[0.22,0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,1.47],[-1.27,0],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,-0.74],[-1.27,-1.27],[-0.22,-0.22],linestyle="-",color="tab:gray")
ax10.plot([0.74,0.74],[-1.27,-1.27],[-0.22,0.22],linestyle="-",color="tab:gray")

#Rhombohedral Brillouin Zone
ax10.plot(BrPts.loc[[0,1]].x,BrPts.loc[[0,1]].y,BrPts.loc[[0,1]].z,linestyle="-",color="k",label="Rhombohedral BZ")
ax10.plot(BrPts.loc[[2,1]].x,BrPts.loc[[2,1]].y,BrPts.loc[[2,1]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[2,4]].x,BrPts.loc[[2,4]].y,BrPts.loc[[2,4]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[5,4]].x,BrPts.loc[[5,4]].y,BrPts.loc[[5,4]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[5,3]].x,BrPts.loc[[5,3]].y,BrPts.loc[[5,3]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[0,3]].x,BrPts.loc[[0,3]].y,BrPts.loc[[0,3]].z,linestyle="-",color="k")

ax10.plot(BrPts.loc[[6,7]].x,BrPts.loc[[6,7]].y,BrPts.loc[[6,7]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[7,8]].x,BrPts.loc[[7,8]].y,BrPts.loc[[7,8]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[8,10]].x,BrPts.loc[[8,10]].y,BrPts.loc[[8,10]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[10,11]].x,BrPts.loc[[10,11]].y,BrPts.loc[[10,11]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[9,11]].x,BrPts.loc[[9,11]].y,BrPts.loc[[9,11]].z,linestyle="-",color="k")
ax10.plot(BrPts.loc[[9,6]].x,BrPts.loc[[9,6]].y,BrPts.loc[[9,6]].z,linestyle="-",color="k")

ax10.plot([B1[0],H2[0]],[B1[1],H2[1]],[B1[2],H2[2]],linestyle="-",color="k")
ax10.plot([B0[0],H1[0]],[B0[1],H1[1]],[B0[2],H1[2]],linestyle="-",color="k")
ax10.plot([B4[0],H7[0]],[B4[1],H7[1]],[B4[2],H7[2]],linestyle="-",color="k")
ax10.plot([B5[0],H8[0]],[B5[1],H8[1]],[B5[2],H8[2]],linestyle="-",color="k")
ax10.plot([H7[0],H8[0]],[H7[1],H8[1]],[H7[2],H8[2]],linestyle="-",color="k")
ax10.plot([H2[0],H6[0]],[H2[1],H6[1]],[H2[2],H6[2]],linestyle="-",color="k")
ax10.plot([H1[0],H5[0]],[H1[1],H5[1]],[H1[2],H5[2]],linestyle="-",color="k")
ax10.plot([H7[0],H4[0]],[H7[1],H4[1]],[H7[2],H4[2]],linestyle="-",color="k")
ax10.plot([H8[0],H3[0]],[H8[1],H3[1]],[H8[2],H3[2]],linestyle="-",color="k")
ax10.plot([B_1[0],H6[0]],[B_1[1],H6[1]],[B_1[2],H6[2]],linestyle="-",color="k")
ax10.plot([B_0[0],H5[0]],[B_0[1],H5[1]],[B_0[2],H5[2]],linestyle="-",color="k")
ax10.plot([B_5[0],H3[0]],[B_5[1],H3[1]],[B_5[2],H3[2]],linestyle="-",color="k")
ax10.plot([B_4[0],H4[0]],[B_4[1],H4[1]],[B_4[2],H4[2]],linestyle="-",color="k")
ax10.plot([B2[0],H10[0]],[B2[1],H10[1]],[B2[2],H10[2]],linestyle="-",color="k")
ax10.plot([B3[0],H9[0]],[B3[1],H9[1]],[B3[2],H9[2]],linestyle="-",color="k")
ax10.plot([H10[0],H12[0]],[H10[1],H12[1]],[H10[2],H12[2]],linestyle="-",color="k")
ax10.plot([H9[0],H11[0]],[H9[1],H11[1]],[H9[2],H11[2]],linestyle="-",color="k")
ax10.plot([B_2[0],H12[0]],[B_2[1],H12[1]],[B_2[2],H12[2]],linestyle="-",color="k")
ax10.plot([B_3[0],H11[0]],[B_3[1],H11[1]],[B_3[2],H11[2]],linestyle="-",color="k")
ax10.plot([H6[0],H5[0]],[H6[1],H5[1]],[H6[2],H5[2]],linestyle="-",color="k")
ax10.plot([H3[0],H11[0]],[H3[1],H11[1]],[H3[2],H11[2]],linestyle="-",color="k")
ax10.plot([H4[0],H12[0]],[H4[1],H12[1]],[H4[2],H12[2]],linestyle="-",color="k")
ax10.plot([H9[0],H1[0]],[H9[1],H1[1]],[H9[2],H1[2]],linestyle="-",color="k")
ax10.plot([H10[0],H2[0]],[H10[1],H2[1]],[H10[2],H2[2]],linestyle="-",color="k")

#MEASURED
#hv150 is kz Gamma (+/)-0.22 (A)
# ax10.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[-0.22,-0.22],linewidth=5,color="red",alpha=0.3) #GM 150eV
# ax10.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.22,0.22],linewidth=5,color="red",alpha=0.3)
# ax10.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[-0.22,-0.22],linewidth=5,color="red",alpha=0.3)#GK 150eV
# ax10.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.22,0.22],linewidth=5,color="red",alpha=0.3)
# #hv163 is kz Gamma 0 (G)
# ax10.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0,0],linewidth=5,color="red",alpha=0.3) #GM 163eV
# ax10.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0,0],linewidth=5,color="red",alpha=0.3)#GK 150eV
# #hv132 is kz Z (T) 0.66 (Z)
# ax10.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.66,0.66],linewidth=5,color="red",alpha=0.3) #GM 132eV
# ax10.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.66,0.66],linewidth=5,color="red",alpha=0.3)#GK 132eV

# ax10.plot([G[0],Mh0[0]],[G[1],Mh0[1]],[0.33,0.33],linewidth=5,color="red",alpha=0.3) #GM 
# ax10.plot([G[0],Kh0[0]],[G[1],Kh0[1]],[0.33,0.33],linewidth=5,color="red",alpha=0.3)#GK 

#Measured TOP VB
#150eV LCO6 FS0
# VBmax150=bh1*0.215+bh3*0.6
# ax10.scatter(VBmax150[0],VBmax150[1],VBmax150[2], color="red",marker="*",s=100)

#Lambrecht hexagonal path
ax10.plot([0.5*bh.loc[1].x,0],[0.5*bh.loc[1].y,0],[0.5*bh.loc[1].z,0],linewidth=5,color="blue",alpha=0.4)#M-G
ax10.plot([0,0.33*bh.loc[1].x+0.33*bh.loc[2].x],[0,0.33*bh.loc[1].y+0.33*bh.loc[2].y],[0,0.33*bh.loc[1].z+0.33*bh.loc[2].z],linewidth=5,color="blue",alpha=0.4)#G-K
ax10.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x,0.5*bh.loc[1].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y,0.5*bh.loc[1].y],[0.33*bh.loc[1].z+0.33*bh.loc[2].z,0.5*bh.loc[1].z],linewidth=5,color="blue",alpha=0.4)#K-M
ax10.plot([0.5*bh.loc[1].x,0.5*bh.loc[1].x+0.5*bh.loc[3].x],[0.5*bh.loc[1].y,0.5*bh.loc[1].y+0.5*bh.loc[3].y],[0,0.22],linewidth=5,color="blue",alpha=0.4)#M-L
ax10.plot([0.5*bh.loc[1].x+0.5*bh.loc[3].x,0.5*bh.loc[3].x],[0.5*bh.loc[1].y+0.5*bh.loc[3].y,0.5*bh.loc[3].y],[0.22,0.22],linewidth=5,color="blue",alpha=0.4)#L-Z
ax10.plot([0.5*bh.loc[3].x,0],[0.5*bh.loc[3].y,0],[0.22,0],linewidth=5,color="blue",alpha=0.4)#Z-G
#ax10.plot([0.5*bh.loc[2].x,0],[0.5*bh.loc[2].y,0],[0.5*bh.loc[2].z,0],linewidth=5,color="blue",alpha=0.4)#M'-G
ax10.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x,0.33*bh.loc[1].x+0.33*bh.loc[2].x+0.5*bh.loc[3].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y,0.33*bh.loc[1].y+0.33*bh.loc[2].y+0.5*bh.loc[3].y],[0,0.22],linewidth=5,color="blue",alpha=0.4)#K-B
ax10.plot([0.33*bh.loc[1].x+0.33*bh.loc[2].x+0.5*bh.loc[3].x,0.5*bh.loc[3].x],[0.33*bh.loc[1].y+0.33*bh.loc[2].y+0.5*bh.loc[3].y,0.5*bh.loc[3].y],[0.22,0.22],linewidth=5,color="blue",alpha=0.4)#B-Z

#Hexagonal Symmetry points
#ax10.scatter(M0[0],M0[1],M0[2],color='b',marker="$M$",s=150)
#ax10.scatter(Kht[0],Kht[1],Kht[2],color='b',marker="$K$",s=150)
ax10.scatter(G_t[0],G_t[1],G_t[2],color='b',marker="$G$",s=150)
#ax10.scatter(L_t[0],L_t[1],L_t[2],color='b',marker="$L$",s=150)
ax10.scatter(G_t[0],G_t[1],0.22,color='b',marker="$A$",s=150)
ax10.scatter(H0_t[0],H0_t[1],H0_t[2],color='b',marker="$H$",s=150)
ax10.scatter(M0_t[0],M0_t[1],M0_t[2],color='b',marker="$L$",s=150)
ax10.scatter(M0_t[0],M0_t[1],0,color='b',marker="$M$",s=150)
ax10.scatter(H0_t[0],H0_t[1],0,color='b',marker="$K$",s=150)





#Lambrecht rhombohedral path
ax10.plot([0.5*br.loc[1].x+0.5*br.loc[2].x+0.5*br.loc[3].x,0],[0.5*br.loc[1].y+0.5*br.loc[2].y+0.5*br.loc[3].y,0],[0.5*br.loc[1].z+0.5*br.loc[2].z+0.5*br.loc[3].z,0],linewidth=5,color="green",alpha=0.4)#G-T
ax10.plot([T_t[0], 0],[T_t[1], 0],[T_t[2], 0],linewidth=5,color="green",alpha=0.4)#G-T
ax10.plot([T_t[0], H0_t[0]],[T_t[1], H0_t[1]],[T_t[2], H0_t[2]],linewidth=5,color="green",alpha=0.4)#T-H
ax10.plot([Lht[0], H0_t[0]],[Lht[1], H0_t[1]],[Lht[2], H0_t[2]],linewidth=5,color="green",alpha=0.4)#H-L
ax10.plot([Lht[0], 0],[Lht[1], 0],[Lht[2], 0],linewidth=5,color="green",alpha=0.4)#L-G
ax10.plot([S0_t[0], 0],[S0_t[1], 0],[S0_t[2], 0],linewidth=5,color="green",alpha=0.4)#G-S
ax10.plot([S0_t[0], F1[0]],[S0_t[1], F1[1]],[S0_t[2], F1[2]],linewidth=5,color="green",alpha=0.4)#S-F
ax10.plot([0, F1[0]],[0, F1[1]],[0, F1[2]],linewidth=5,color="green",alpha=0.4)#F-G

#Rhombohedral SYmmetry points
ax10.scatter(Lht[0],Lht[1],Lht[2],color='g',marker="$L$",s=150)
ax10.scatter(G_t[0],G_t[1],G_t[2],color='g',marker="$G$",s=150)
ax10.scatter(F1[0],F1[1],F1[2],color='g',marker="$F$",s=150)
ax10.scatter(H0_t[0],H0_t[1],H0_t[2],color='g',marker="$H$",s=150)
ax10.scatter(T_t[0],T_t[1],T_t[2],color='g',marker="$T$",s=150)




#Calculated TOP VB
# VBmaxTheory=np.array([0.456,0,0.56])
# ax10.scatter(VBmaxTheory[0],VBmaxTheory[1],VBmaxTheory[2],color="green",marker="*",s=100)

ax10.xaxis.pane.fill = False
ax10.yaxis.pane.fill = False
ax10.zaxis.pane.fill = False
ax10.set_xlim(-2, 2); ax10.set_ylim(-2, 2); ax10.set_zlim(-2, 2);
ax10.view_init(30, -60)
ax10.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax10.grid(False)
#ax10.legend()
plt.show()


#################### THE END ########################