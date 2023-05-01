#!/usr/bin/env python
# coding: utf-8


from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



#!Inserire il numero di test desiderato tra 1 e 3!
test=
TEST_COVARIANZA=True
PLOTS=True




Nbins=200
Nmisure=10000
misure=[]



#!Cambiare percorso file con quanto nel vostro computatore!
for i in np.arange(Nmisure)+1:
    fname=f'Scaricati/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'

    file=fits.open(fname)
    table=file[1].data.copy()
    misure.append(table['XI0'])
    if i==1:
        scale=table['SCALE']
    del table
    file.close()
    
misure=np.asarray(misure).transpose()



#Covarianza calcolata
media_xi=np.mean(misure, axis=1)
cov_xi=np.cov(misure)




# Grafico della matrice di covarianza trovata
plt.matshow(cov_xi)
plt.title('Grafico della matrice di covarianza misurata')
cbar = plt.colorbar(orientation="vertical", pad=0.01)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()




#Matrice di correlazione
corr_xi=np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        corr_xi[i,j]=cov_xi[i,j]/(cov_xi[i,i]*cov_xi[j,j])**0.5





if test==1:
    sigs = [0.02, 0.02, 0.02]
    ls = [25, 50, 75]
elif test==2:
    sigs = [0.02, 0.01, 0.005]
    ls = [50, 50, 50]
else:
    sigs = [0.02, 0.01, 0.005]
    ls = [5, 5, 5]




#Matrice di covarianza
def covf(r1, r2, sigma, h):
    return sigma**2.*np.exp(-(r1-r2)**2./(2.*h**2.))




cov_tr=np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov_tr[i,j]=covf(scale[i], scale[j], sigs[0], ls[0])





#grafico covarianza teorica
plt.matshow(cov_tr)
plt.title('Grafico della matrice di covarianza teorica')
cbar = plt.colorbar(orientation="vertical", pad=0.01)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()




residui_norm=np.zeros_like(cov_tr)
for i in range(Nbins):
    for j in range(Nbins):
        R=cov_tr[i,j]**2./(np.sqrt(cov_tr[i,i]*cov_tr[j,j])**2.)
        residui_norm[i,j]=(cov_tr[i,j]-cov_xi[i,j])*np.sqrt((Nmisure-1.)/((1.+R)*cov_tr[i,i]*cov_tr[j,j]))
        
rms_deviazione=np.std(residui_norm.reshape(Nbins**2))

print(f"deviazione rms dei residui normalizzati:{rms_deviazione}")

if rms_deviazione<1.1:
    print("!VERIFICATA!")
else:
    print("!FALLITA!")




#grafico residui
plt.matshow(cov_tr-cov_xi)
plt.title('Grafico dei residui')
cbar = plt.colorbar(orientation="vertical", pad=0.01)
cbar.set_label(r'$ C^{\xi}_{N}$')
plt.show()







