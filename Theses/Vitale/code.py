#importa il video
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof

#load video from file
cap=cv2.VideoCapture('pendolo2fine.mp4') #importa il video
#print('formato video=',type(cap))

#conta i frame
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ti da il numero di frame
print('frame totali=' , frame_count)

#conta le dimensioni dei frame
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print('dimensioni=',width,height)

#conta i fps
fps = int(cap.get(cv2.CAP_PROP_FPS))
#print('fps=',fps)

print('VIDEO IMPORTATO')

###################################################
###################################################
###################################################


#creo tensore
video_tensor=np.zeros((300,512,512,3),dtype='float')
video_tensor_fake=np.zeros((frame_count,height,width,3),dtype='float')
count=0
while cap.isOpened(): #è una vaiabile booleana che è vera se il video è aperto tipo
    ret,frame=cap.read() #cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
    #in pratice ret diventa true se trova il frame di dopo mentre frame diventa il frame 
    if ret is True:
        video_tensor_fake[count,:,:,:]=frame #modifichi video_tensor[x,:,:,:]
        count=count + 1
    else:
        break
video_tensor=video_tensor_fake[frame_count-301:frame_count, int((height-512)/2) : -int((height-512)/2) ,int((width-512)/2):-int((width-512)/2),:]
video=video_tensor
print(np.shape(video))
print('TENSORE CREATO')

###################################################
###################################################
###################################################
print("Più strati? (lasciare vuoto per dire di no):")
multistrato = bool(input())
print("Gauss = pieno || Laplace = vuoto")
laporgauss=bool(input())
print('Butt = Pieno || Ideal = Vuoto')
buttt=bool(input())
#print("Vuoi aggiungere la media?")
#print('Non dovrebbe mai servire, con Gauss fa guai.')
#print('Lascia vuoto per dire di no.')
sommed=False #bool(input())
if (multistrato == False):
  print("A che livello spaziale vuoi lavorare? (da 1 a 4)")
  livello = int(input())
#print('Titolo video finale:')
#stringa=input()
if (multistrato == True): 
  stringa1='multi_'
else:
  stringa1= 'liv'+str(livello)+'_'
if (laporgauss == True):
  stringa2 = 'gauss_'
else:
  stringa2= 'lap_'
if (buttt == True):
  stringa3='butt_'
else: 
  stringa3='ideal_'
print('Frequenza inferiore?:')
low=float(input())
print('Frequenza superiore?:')
high=float(input())
stringa4='('+str(low)+'-'+str(high)+')_'
print('Alpha:')
amplificazione=float(input())
stringa5='alpha='+str(amplificazione)
stringa=stringa1+stringa2+stringa3+stringa4+stringa5

###################################################
###################################################
###################################################

#A STO PUNTO CREO PER OGNI FRAME UN PYRDOWN PER AVERE gaussiana PIRAMIDE.
alt=np.size(video_tensor[1,1,:,1])
larg=np.size(video_tensor[1,1,:,1])
frame_count=np.size(video_tensor[:,1,1,1,])

vg1=np.zeros((frame_count,int(alt/2),int(larg/2),3),dtype='float')
for i in range(300):
    vg1[i,:,:,:]=cv2.pyrDown(video_tensor[i,:,:,:])

vg2=np.zeros((frame_count,int(alt/4),int(larg/4),3),dtype='float')
for i in range(300):
    vg2[i,:,:,:]=cv2.pyrDown(vg1[i,:,:,:])

vg3=np.zeros((frame_count,int(alt/8),int(larg/8),3),dtype='float')
for i in range(300):
    vg3[i,:,:,:]=cv2.pyrDown(vg2[i,:,:,:])

vg4=np.zeros((frame_count,int(alt/16),int(larg/16),3),dtype='float')
for i in range(300):
    vg4[i,:,:,:]=cv2.pyrDown(vg3[i,:,:,:])

vg5=np.zeros((frame_count,int(alt/32),int(larg/32),3),dtype='float')
for i in range(300):
    vg5[i,:,:,:]=cv2.pyrDown(vg4[i,:,:,:])

piramidegaussiana=[video_tensor,vg1,vg2,vg3,vg4,vg5]
#è una lista di tensori di 4 dimensioni quindi ci accedi così lista[:][:,:,:,:]
print('PIRAMIDA GAUSS FINITA')

###################################################
###################################################
###################################################

#qui invece creo la piramide laplaciana 
pirlaplace=[np.zeros((301,512,512,3))]
pirlaplace.append(np.zeros((301,256,256,3)))
pirlaplace.append(np.zeros((301,128,128,3)))
pirlaplace.append(np.zeros((301,64,64,3)))
pirlaplace.append(np.zeros((301,32,32,3)))


#devo fare il ciclo al contrario perché mi serve quella più resoluta per definire.
for j in range(300):
  for i in range(5,0,-1):
    gpext=cv2.pyrUp(piramidegaussiana[i][j,:,:,:])
    pirlaplace[i-1][j]=cv2.subtract(piramidegaussiana[i-1][j],gpext) 

print('PIRAMIDE LAPLACE FINITA')

###################################################
###################################################
###################################################

#decido se si lavor con gauss o con laplace
if (laporgauss):
  for i in range(len(pirlaplace)):
    pirlaplace[i]=piramidegaussiana[i]

###################################################
###################################################
###################################################

#tagliare ultimo frame #Da far girare una sola volta e non più di una!
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import fftpack
%matplotlib inline

for j in range(5):
    filmino=np.delete(pirlaplace[j][:,:,:,:],-1,0)
    if (j==0):
        film=[filmino]
    else:
        film.append(filmino)
print('TAGLIO ULTIMO FRAME')

###################################################
###################################################
###################################################
#qui comincia una differenzazione dal singolo strato e dal multi strato

from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import fftpack
%matplotlib inline
import datetime

######## AGGIUNGO PER BUTTER ##########
if (buttt == True):
  if (multistrato == True):
    for i in range(1,5):                  
      fs=filmino[i].shape[0]
      fps=30
      t=np.arange(0,10,1/fps) 
      media=np.mean(film[i][:,:,:,:],axis=0)

      if (i==1):                       
        listamedia=[media]
      else:
        listamedia.append(media)
      film_scalato=np.subtract(film[i],media) ##lo definisco ogni volta e poi lo "perdo"
      assefreq = fftpack.fftfreq(film[i].shape[0], d=1.0 / fps) 
      tensore_traformato=film_scalato #che contiene il doppio spettro
      if (i==1):                        
        listatensori=[tensore_traformato]
      else:
        listatensori.append(tensore_traformato)
  else:
    fs=filmino[livello].shape[0]
    fps=30
    t=np.arange(0,10,1/fps)     
    media=np.mean(film[livello][:,:,:,:],axis=0)
    listamedia=[media]
    film_scalato=np.subtract(film[livello],media)
    assefreq = fftpack.fftfreq(film[livello].shape[0], d=1.0 / fps) 
    tensore_traformato=film_scalato #che contiene il doppio spettro  
    listatensori=[tensore_traformato]

############ FINE AGGIUNTE ################

if (buttt == False):
  if (multistrato == True):
    for i in range(1,5):                  
      fs=filmino[i].shape[0]
      fps=30
      t=np.arange(0,10,1/fps) 
      media=np.mean(film[i][:,:,:,:],axis=0)

      if (i==1):                        
        listamedia=[media]
      else:
        listamedia.append(media)
      film_scalato=np.subtract(film[i],media) ##lo definisco ogni volta e poi lo "perdo"
      assefreq = fftpack.fftfreq(film[i].shape[0], d=1.0 / fps) 
      tensore_traformato=fft(film_scalato,axis=0) #che contiene il doppio spettro
      if (i==1):                        
        listatensori=[tensore_traformato]
      else:
        listatensori.append(tensore_traformato)
    #print("Mi aspetto che sto numero sia uguale a:",4)
    #print(len(listatensori))
  else:
    fs=filmino[livello].shape[0]
    fps=30
    t=np.arange(0,10,1/fps)     
    media=np.mean(film[livello][:,:,:,:],axis=0)
    listamedia=[media]
    film_scalato=np.subtract(film[livello],media)
    assefreq = fftpack.fftfreq(film[livello].shape[0], d=1.0 / fps) 
    tensore_traformato=fft(film_scalato,axis=0) #che contiene il doppio spettro  
    listatensori=[tensore_traformato]
  perplot=listatensori
  print('TRASFORMATO')
###################################################
###################################################
###################################################

if (buttt == True):
  from scipy.signal import butter, lfilter

  def butter_bandpass(lowcut, highcut, fs, order=5):
      nyq = 0.5 * fs #definisce quelaa di nynquist come la metà del campionamento 
      lowb = lowcut / nyq #esprime la freq in come fraz di freq di nynquist
      highb = highcut / nyq #idem
      b, a = butter(order, [lowb, highb], btype='band') #qui sembra fare il passa-banda-butter semplice.
      return b, a
  #alla fine non è altro che la funzione di scipy con il riscalo delle frequenze rispetto a quella di nyq

  def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
      b, a = butter_bandpass(lowcut, highcut, fs, order=order)
      y = lfilter(b, a, data) #applica veramente il filtro
      return y
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.signal import freqz

  fs = 30.0 #campionamento dei dati
  lowcut = low #dove voglio tagliare le frequenze
  highcut = high #dove le voglio tagliare ancora
  print('DEFINITO BUTTER')

###################################################
###################################################
###################################################
if (buttt==True):
  if (multistrato==True):
    for j in range(4): 
      tensore_filtrato=np.copy(listatensori[j]) #lo metto per la size
      for i in range(np.shape(tensore_filtrato[0,:,0,0])[0]):
        for l in range(np.shape(tensore_filtrato[0,0,:,0])[0]):
          for k in range(3):
            tensore_filtrato[:,i,l,k]=butter_bandpass_filter(listatensori[j][:,i,l,k], lowcut, highcut, fs, order=6) 
      if (j==0):
        lista_tensori_filt=[tensore_filtrato]
      else:
        lista_tensori_filt.append(tensore_filtrato)
  
  else: 
    tensore_filtrato=np.copy(listatensori[0])
    for i in range(np.shape(tensore_filtrato[0,:,0,0])[0]):
      for l in range(np.shape(tensore_filtrato[0,0,:,0])[0]):
        for k in range(3):
          tensore_filtrato[:,i,l,k]=butter_bandpass_filter(listatensori[0][:,i,l,k], lowcut, highcut, fs, order=6) 
          lista_tensori_filt=[tensore_filtrato]
  salvato=lista_tensori_filt
  print(type(salvato),'contiene output di butter pre ampli')
  
  print('BUTTER FINITO')
###################################################
###################################################
###################################################
#CONTINUA PER IDEAL
#ideal passband filter
#low=0.4 valore faccia  #4.0 valore per cut
#high=4. valore faccia  #6. valore per cut
if (buttt == False):
  bound_low = (np.abs(assefreq - low)).argmin()
  bound_high = (np.abs(assefreq - high)).argmin()
  if (multistrato==True):
    for j in range(4):        #QUI USO FINO A 4 DATO CHE NON HO RIDEFINITO LA LISTA
      tensore_filtrato=np.copy(listatensori[j])
      tensore_filtrato[:bound_low,:,:,:] = 0
      tensore_filtrato[bound_high:-bound_high,:,:,:] = 0 
      tensore_filtrato[-bound_low:,:,:,:] = 0
      if (j==0):
        lista_tensori_filt=[tensore_filtrato]
      else:
        lista_tensori_filt.append(tensore_filtrato)

  else:
    tensore_filtrato=np.copy(listatensori[0])
    tensore_filtrato[:bound_low,:,:,:] = 0
    tensore_filtrato[bound_high:-bound_high,:,:,:] = 0
    tensore_filtrato[-bound_low:,:,:,:] = 0
    lista_tensori_filt=[tensore_filtrato]
  print('PASSBAND FINITO')
###################################################
###################################################
###################################################

  #ANTITRASFORMO 
if (buttt==False):  
  if (multistrato==True):
    for i in range(4): 
      segnale_filtrato=fftpack.ifft(lista_tensori_filt[i],axis=0)
      lista_tensori_filt[i]=segnale_filtrato
  else:
    segnale_filtrato=fftpack.ifft(lista_tensori_filt[0],axis=0)
    lista_tensori_filt[0]=segnale_filtrato
  print('ANTITRASFORMATO')
###################################################
###################################################
###################################################
if (buttt==False):  
  if (multistrato==True):
    print('siamo qui')

#qui si ricongiungono le strade di butt e ideal 
#amplifica il segnale

    matamp0=np.zeros((256,256,3))
    matamp1=np.zeros((128,128,3))
    matamp2=np.zeros((64,64,3))
    matamp3=np.zeros((32,32,3))
    #qui ho il vettore con lo spettro (0.5/300)*abs(listatensori[0][0:150,:,:,:])
    #definisco media spettro per tutti punti
    mediaspettro=[np.mean(abs(listatensori[0][0:150,:,:,:]),axis=0)]
    mediaspettro.append(np.mean(abs(listatensori[1][0:150,:,:,:]),axis=0))
    mediaspettro.append(np.mean(abs(listatensori[2][0:150,:,:,:]),axis=0))
    mediaspettro.append(np.mean(abs(listatensori[3][0:150,:,:,:]),axis=0))
    sigg=5.0
    for i in range(256):
      for j in range(256):
        for k in range(3):
          if (np.any(abs(listatensori[0][0:150,i,j,k])>(mediaspettro[0][i,j,k]+sigg*np.std(abs(listatensori[0][0:150,i,j,k]))))): 
            matamp0[i,j,k]=amplificazione
    for i in range(128):
      for j in range(128):
        for k in range(3):
          if (np.any(abs(listatensori[1][0:150,i,j,k])>(mediaspettro[1][i,j,k]+sigg*np.std(abs(listatensori[1][0:150,i,j,k]))))):
            matamp1[i,j,k]=amplificazione*0.8
    for i in range(64):
      for j in range(64):
        for k in range(3):
          if (np.any(abs(listatensori[2][0:150,i,j,k])>(mediaspettro[2][i,j,k]+sigg*np.std(abs(listatensori[2][0:150,i,j,k]))))):
            matamp2[i,j,k]=amplificazione*0.6
    for i in range(32):
      for j in range(32):
        for k in range(3):
          if (np.any(abs(listatensori[3][0:150,i,j,k])>(mediaspettro[3][i,j,k]+sigg*np.std(abs(listatensori[3][0:150,i,j,k]))))):
            matamp3[i,j,k]=amplificazione*0.4
    print('e adesso qui')        

    lista_tensori_filt[0]=matamp0*lista_tensori_filt[0] #difficilissima sta moltiplicazione    
    lista_tensori_filt[1]=matamp1*lista_tensori_filt[1]    
    lista_tensori_filt[2]=matamp2*lista_tensori_filt[2]    
    lista_tensori_filt[3]=matamp3*lista_tensori_filt[3]    
    print('e speriamo che la moltiplicazione abbia funzio')
  else:  
    lista_tensori_filt[0] = amplificazione*lista_tensori_filt[0]



#if (multistrato==True):  

#  ampli0=amplificazione
#  ampli1=amplificazione
#  ampli2=amplificazione
#  ampli3=amplificazione
#  lista_tensori_filt[0] = ampli0*lista_tensori_filt[0]
#  lista_tensori_filt[1] = ampli1*lista_tensori_filt[1]
#  lista_tensori_filt[2] = ampli2*lista_tensori_filt[2]
#  lista_tensori_filt[3] = ampli3*lista_tensori_filt[3]

#else:  
#  lista_tensori_filt[0] = amplificazione*lista_tensori_filt[0]

#riaggiungere media 
if (sommed==True):
  if (multistrato==True):
    for i in range(4):
      lista_tensori_filt[i]=np.add(lista_tensori_filt[i],listamedia[i])
      #Attento controlla la lista medie se ha indicizzazione da 1 a 4 o da 1 a 5
  else:
    lista_tensori_filt[0]=np.add(lista_tensori_filt[0],listamedia[0])
print('AMPLIFICATO')
###################################################
###################################################
###################################################

#reconstract video from original video and gaussian video
if (multistrato==True):
  final_video=np.zeros((300,512,512,3))
  for i in range(300): 
    frame_finale=cv2.pyrUp(np.real(lista_tensori_filt[0][i,:,:,:]))
    frame_finale=frame_finale+cv2.pyrUp(cv2.pyrUp(np.real(lista_tensori_filt[1][i,:,:,:])))
    frame_finale=frame_finale+cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(np.real(lista_tensori_filt[2][i,:,:,:]))))
    frame_finale=frame_finale+cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(np.real(lista_tensori_filt[3][i,:,:,:]))))) 
    frame_finale = frame_finale + video_tensor[i,:,:,:]  #fa una somma tra originale e img 
    final_video[i]=frame_finale


else:  
  final_video=np.zeros((300,512,512,3))
  for i in range(300): 
    frame_finale=np.real(lista_tensori_filt[0][i,:,:,:])
    for j in range(livello):
        frame_finale = cv2.pyrUp(frame_finale)
    frame_finale = frame_finale + video_tensor[i,:,:,:]  #fa una somma tra originale e img 
    final_video[i]=frame_finale #sta aggiungendo frame per frame
print('VIDEO CREATO')

###################################################
###################################################
###################################################

#solo qui si ricongiungono strade tra multi e livello
#create and save the video
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
[height,width]=final_video[0].shape[0:2]
writer = cv2.VideoWriter('pendolometa_'+stringa+'sig5.0_zeros_'+'.avi', fourcc, 30, (width, height), 1)
#1 sarebbe True che significa che è colorato
for i in range(0,final_video.shape[0]):
    writer.write(cv2.convertScaleAbs(final_video[i]))
writer.release()
print('VIDEO SALVATO')

