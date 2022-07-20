from skimage import io,data,filters,segmentation,measure,morphology,color,feature,draw
from skimage.util import img_as_ubyte
from scipy import ndimage,optimize,signal,interpolate
import numpy as np
from math import pi,sqrt,sin,cos,ceil,floor,degrees
import cv2
import json
import copy
from multiprocessing import Pool
import warnings
from functools import partial

warnings.filterwarnings('ignore')

#function to fulfill mask
def fitrec(msize,nsize,tx,ty):
    tx0=tx[:]
    ty0=ty[:]
    tx0[2],tx0[3]=tx0[3],tx0[2]
    ty0[2], ty0[3] = ty0[3], ty0[2]
    maskrec=draw.polygon2mask([msize,nsize], np.vstack((ty0,tx0)).T).astype(float)
    '''
    tx=np.ceil(tx).astype(int)
    ty=np.ceil(ty).astype(int)
    maskrec=np.zeros((msize,nsize),dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        
    for p in range(min(ty),max(ty)):
        for q in range(min(tx),max(tx)):
            vec1=np.array([tx[0]-q,ty[0]-p])
            vec2=np.array([tx[1]-q,ty[1]-p])
            vec3=np.array([tx[2]-q,ty[2]-p])
            vec4=np.array([tx[3]-q,ty[3]-p])
            sita1=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
            sita2=np.arccos(np.dot(vec1,vec3)/(np.linalg.norm(vec1)*np.linalg.norm(vec3)))
            sita3=np.arccos(np.dot(vec2,vec3)/(np.linalg.norm(vec2)*np.linalg.norm(vec3)))
            sita4=np.arccos(np.dot(vec2,vec4)/(np.linalg.norm(vec2)*np.linalg.norm(vec4)))
            sita5=np.arccos(np.dot(vec3,vec4)/(np.linalg.norm(vec3)*np.linalg.norm(vec4)))
            if (sita1+sita2+sita3<2*pi+0.02 and sita1+sita2+sita3>2*pi-0.02) or \
                (sita3+sita4+sita5<2*pi+0.02 and sita3+sita4+sita5>2*pi-0.02) or \
                (np.linalg.norm(vec1)*np.linalg.norm(vec2)*np.linalg.norm(vec3)*np.linalg.norm(vec4)==0):
                maskrec[p,q]=1
    maskrec_close = cv2.morphologyEx(maskrec, cv2.MORPH_CLOSE, kernel,iterations=1) 
    '''
    return maskrec

def fitrec_tr(msize,nsize,tx,ty):
    maskrec = draw.polygon2mask([msize, nsize], np.vstack((ty, tx)).T).astype(float)

    '''
    tx=np.ceil(tx).astype(int)
    ty=np.ceil(ty).astype(int)
    maskrec=np.zeros((msize,nsize),dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        
    for p in range(min(ty),max(ty)):
        for q in range(min(tx),max(tx)):
            vec1=np.array([tx[0]-q,ty[0]-p])
            vec2=np.array([tx[1]-q,ty[1]-p])
            vec3=np.array([tx[2]-q,ty[2]-p])
            sita1=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
            sita2=np.arccos(np.dot(vec1,vec3)/(np.linalg.norm(vec1)*np.linalg.norm(vec3)))
            sita3=np.arccos(np.dot(vec2,vec3)/(np.linalg.norm(vec2)*np.linalg.norm(vec3)))
            if (sita1+sita2+sita3<2*pi+0.02 and sita1+sita2+sita3>2*pi-0.02) or \
                (np.linalg.norm(vec1)*np.linalg.norm(vec2)*np.linalg.norm(vec3)==0):
                maskrec[p,q]=1
    maskrec_close = cv2.morphologyEx(maskrec, cv2.MORPH_CLOSE, kernel,iterations=1) 
    '''
    return maskrec

#calculate the height of roof                  
def fitmod(mask,ptz,lsq,mask_fit):
    maskz=copy.deepcopy(mask)
    ptz=np.mat(ptz)
    lsq=np.mat(lsq)
    try:
        plane=np.linalg.inv((lsq.T)*lsq)*(lsq.T)*ptz.T
    except:
        plane = (lsq.T)*ptz.T
    (m,n)=np.shape(maskz)
    maska=np.ones((m,n),dtype=float)
    maskb=np.ones((m,n),dtype=float)
    maska=np.dot(np.array(range(m)).reshape(m,1),np.ones((1,n)))
    maskb=np.dot(np.ones((m,1)),np.array(range(n)).reshape(1,n))
    #for k in range(m):
    #    maska[k,:]=maska[k,:]*k
    #for k in range(n):
    #    maskb[:,k]=maskb[:,k]*k
    plane=np.array(plane)
    maskab=plane[0]*maskb+plane[1]*maska+plane[2]
    maskab[np.where(mask_fit==0)]=0
    maskz[np.where(maskab>0)]=maskab[np.where(maskab>0)]
    
    #(pta,ptb)=np.where(mask_fit==1)
    #for k in range(len(pta)):
    #    maskz[pta[k],ptb[k]]=plane[0]*ptb[k]+plane[1]*pta[k]+plane[2]
    return maskz

def fitinterp(mask,ptz,ptn,mask_fit):
    maskz = copy.deepcopy(mask)
    (msize, nsize) = np.shape(maskz)
    #f_interp = interpolate.interp2d(ptn[:,0], ptn[:,1], ptz, kind='linear')
    if abs(ptn[0,0]-ptn[1,0])<1 and abs(ptn[0,1]-ptn[1,1])<1:
        return maskz
    else:
        f_interp = interpolate.LinearNDInterpolator(list(ptn), ptz)
        gridx = np.linspace(0, nsize, nsize)
        gridy = np.linspace(0, msize, msize)
        gridx, gridy = np.meshgrid(gridx, gridy)
        gridz = f_interp(gridx, gridy)
        maskout = mask_fit * gridz
        maskz[np.where(mask_fit > 0)] = maskout[np.where(mask_fit > 0)]

        return maskz

def fit_flat(hight0,dzeave,mask_ht,mask_rec,bmask):
    para3d_f=[]
    for zeave in hight0+dzeave:
        rmse_f=np.sum(((mask_ht-mask_rec*zeave)**2)*bmask)/np.sum(mask_rec*bmask)
        para3d_f.append(np.array([zeave,zeave,0,0,0,rmse_f]))
    mf_idx=np.argmin(np.array(para3d_f), axis=0)[5]
    return para3d_f,mf_idx
        
def fit_gable(hight0,dzeave,dzridge,del_ht,rec,mask_ht,mask_rec,bmask):
    (msize,nsize)=np.shape(bmask)
    gxyp=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
    gxyv=np.array([(rec[8]+rec[12])/2,(rec[9]+rec[13])/2,(rec[10]+rec[14])/2,(rec[11]+rec[15])/2])
    para3d_g_p=[]
    para3d_g_v=[]
    #direction of parallel
    ptnp1=np.array([[rec[8],rec[9]],[gxyp[0],gxyp[1]],[rec[12],rec[13]],[gxyp[2],gxyp[3]]])
    lsqp1=np.hstack((ptnp1,np.ones((4,1),dtype=float)))
    maskp1=fitrec(msize,nsize,ptnp1[:,0],ptnp1[:,1])
    ptnp2=np.array([[rec[10],rec[11]],[gxyp[0],gxyp[1]],[rec[14],rec[15]],[gxyp[2],gxyp[3]]])
    lsqp2=np.hstack((ptnp2,np.ones((4,1),dtype=float)))
    maskp2=mask_rec-maskp1
    #direction of vertical
    ptnv1=np.array([[rec[8],rec[9]],[gxyv[0],gxyv[1]],[rec[10],rec[11]],[gxyv[2],gxyv[3]]])
    lsqv1=np.hstack((ptnv1,np.ones((4,1),dtype=float)))
    maskv1=fitrec(msize,nsize,ptnv1[:,0],ptnv1[:,1])
    ptnv2=np.array([[rec[12],rec[13]],[gxyv[0],gxyv[1]],[rec[14],rec[15]],[gxyv[2],gxyv[3]]])
    lsqv2=np.hstack((ptnv2,np.ones((4,1),dtype=float)))
    maskv2=mask_rec-maskv1

    for zeave in hight0+dzeave-del_ht:
        for zridge in zeave+dzridge:
            ptz=np.array([zeave,zridge,zeave,zridge]).T
            #parallel
            maskpz=np.zeros((msize,nsize),dtype=float)
            maskpz=fitmod(maskpz,ptz,lsqp1,maskp1)
            maskpz=fitmod(maskpz,ptz,lsqp2,maskp2)
            mask_diff=mask_ht-mask_rec*maskpz
            mask_diff[np.where(np.absolute(mask_diff)>10)]=0
            rmse_g=np.sum(((mask_diff)**2)*bmask)/np.sum(mask_rec*bmask)
            para3d_g_p.append(np.array([zeave,zridge,0,0,1,rmse_g]))
            #vertical
            maskvz=np.zeros((msize,nsize),dtype=float)
            maskvz=fitmod(maskvz,ptz,lsqv1,maskv1)
            maskvz=fitmod(maskvz,ptz,lsqv2,maskv2)
            mask_diff=mask_ht-mask_rec*maskvz
            mask_diff[np.where(np.absolute(mask_diff)>10)]=0
            rmse_g=np.sum(((mask_diff)**2)*bmask)/np.sum(mask_rec*bmask)
            para3d_g_v.append(np.array([zeave,zridge,0,0,2,rmse_g]))
    para3d_g=para3d_g_p+para3d_g_v
    mg_idx=np.argmin(np.array(para3d_g), axis=0)[5]
    return para3d_g,mg_idx

def fit_hip(hight0,para3d_g,mg_idx,rec,mask_ht,mask_rec,bmask):
    hip01, hip02=rec[4]/4, rec[5]/4
    dhip1, dhip2=np.arange(-rec[4]/8,rec[4]/8,max(2,rec[4]/16)), np.arange(-rec[5]/8,rec[5]/8,max(2,rec[5]/16))
    dhip0=np.arange(-1,1,0.4)
    if len(dhip2)<=1:
        dhip2=np.arange(-1,1,1)
    if len(dhip1)<=1:
        dhip1=np.arange(-1,1,1)
    dzeave_h=np.arange(-0.5,0.5,0.3)
    dzridge_h=np.arange(-0.5,0.5,0.3)
    zeave_h=para3d_g[mg_idx][0]
    zridge_h=para3d_g[mg_idx][1]
    para3d_h_p=[]
    para3d_h_v=[]
    para3d_h=[]
    
    #direction of parallel
    #rough search (reduce time cost)
    dirflag=1
    for hip in hip01+dhip1:
        rt=hip/rec[4]
        if hip<rec[4]/8 or hip>3*rec[4]/8:
            continue
        para3d_ht=calculate_hip_height(rec,rt,hip,zeave_h,zridge_h,dzeave_h,dzridge_h,dirflag,mask_ht,mask_rec,bmask)
        para3d_h_p.extend(para3d_ht)
    mhp_idx=np.argmin(np.array(para3d_h_p), axis=0)[5]
    hip01=para3d_h_p[mhp_idx][2]
    for hip in hip01+dhip0:
        rt=hip/rec[4]
        if hip<rec[4]/8 or hip>3*rec[4]/8:
            continue
        para3d_ht=calculate_hip_height(rec,rt,hip,zeave_h,zridge_h,dzeave_h,dzridge_h,dirflag,mask_ht,mask_rec,bmask)
        para3d_h.extend(para3d_ht)
    dirflag=2
    for hip in hip02+dhip2:
        rt=hip/rec[5]
        if hip<rec[5]/8 or hip>3*rec[5]/8:
            continue
        para3d_ht=calculate_hip_height(rec,rt,hip,zeave_h,zridge_h,dzeave_h,dzridge_h,dirflag,mask_ht,mask_rec,bmask)
        para3d_h_v.extend(para3d_ht)
    mhv_idx=np.argmin(np.array(para3d_h_v), axis=0)[5]
    hip02=para3d_h_v[mhv_idx][2]
    for hip in hip02+dhip0:
        rt=hip/rec[5]
        if hip<rec[5]/8 or hip>3*rec[5]/8:
            continue
        para3d_ht=calculate_hip_height(rec,rt,hip,zeave_h,zridge_h,dzeave_h,dzridge_h,dirflag,mask_ht,mask_rec,bmask)
        para3d_h.extend(para3d_ht)
    mh_idx=np.argmin(np.array(para3d_h), axis=0)[5]
    return para3d_h,mh_idx
    
    
def calculate_hip_height(rec,rt,hip,hight0,zridge_h,dzeave_h,dzridge_h,dirflag,mask_ht,mask_rec,bmask):
    para3d_h=[]
    (msize,nsize)=np.shape(bmask)
    gxy0=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
    gxy=np.array([gxy0[0]*(1-rt)+gxy0[2]*rt,gxy0[1]*(1-rt)+gxy0[3]*rt,\
                  gxy0[0]*rt+gxy0[2]*(1-rt),gxy0[1]*rt+gxy0[3]*(1-rt)])
    if dirflag==1:
        ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]],[gxy[2],gxy[3]]])
        ptn2=np.array([[rec[10],rec[11]],[gxy[0],gxy[1]],[rec[14],rec[15]],[gxy[2],gxy[3]]])
        ptn3=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]]])
        ptn4=np.array([[rec[12],rec[13]],[gxy[2],gxy[3]],[rec[14],rec[15]]])
    else:
        ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]],[gxy[2],gxy[3]]])
        ptn2=np.array([[rec[12],rec[13]],[gxy[0],gxy[1]],[rec[14],rec[15]],[gxy[2],gxy[3]]])
        ptn3=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]]])
        ptn4=np.array([[rec[10],rec[11]],[gxy[2],gxy[3]],[rec[14],rec[15]]])
    lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
    mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
    lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
    mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
    lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
    mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
    lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
    mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])
    for zeave in hight0+dzeave_h:
        for zridge in zridge_h+dzridge_h:
            if zridge-zeave<0.5:
                continue
            ptz=np.array([zeave,zridge,zeave,zridge]).T
            ptz_tr=np.array([zeave,zridge,zeave]).T
            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz,lsq1,mask1)
            maskz=fitmod(maskz,ptz,lsq2,mask2)
            maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
            maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
            mask_diff=mask_ht-mask_rec*maskz
            mask_diff[np.where(np.absolute(mask_diff)>10)]=0
            rmse_h=np.sum(((mask_diff)**2)*bmask)/np.sum(mask_rec*bmask)
            para3d_h.append(np.array([zeave,zridge,hip,0,dirflag,rmse_h]))
    return para3d_h

def fit_pyramid(hight0,dzeave,dzridge,del_ht,rec,mask_ht,mask_rec,bmask):
    para3d_p=[]
    (msize,nsize)=np.shape(bmask)
    gxy=np.array([np.mean(rec[8::2]),np.mean(rec[9::2])])
    ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]]])
    ptn2=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]]])
    ptn3=np.array([[rec[10],rec[11]],[gxy[0],gxy[1]],[rec[14],rec[15]]])
    ptn4=np.array([[rec[12],rec[13]],[gxy[0],gxy[1]],[rec[14],rec[15]]])
    lsq1=np.hstack((ptn1,np.ones((3,1),dtype=float)))
    mask1=fitrec_tr(msize,nsize,ptn3[:,0],ptn1[:,1])
    lsq2=np.hstack((ptn2,np.ones((3,1),dtype=float)))
    mask2=fitrec_tr(msize,nsize,ptn3[:,0],ptn2[:,1])
    lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
    mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
    lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
    mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])
    for zeave in hight0+dzeave-del_ht:
        for zridge in zeave+dzridge:
            ptz_tr=np.array([zeave,zridge,zeave]).T
            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz_tr,lsq1,mask1)
            maskz=fitmod(maskz,ptz_tr,lsq2,mask2)
            maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
            maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
            mask_diff=mask_ht-mask_rec*maskz
            mask_diff[np.where(np.absolute(mask_diff)>10)]=0
            rmse_p=np.sum(((mask_diff)**2)*bmask)/np.sum(mask_rec*bmask)
            para3d_p.append(np.array([zeave,zridge,0,0,0,rmse_p]))
    mp_idx=np.argmin(np.array(para3d_p), axis=0)[5]
    return para3d_p,mp_idx

def fit_mansard(hight0,para3d_g,mg_idx,rec,mask_ht,mask_rec,bmask):
    minlw=min(rec[4],rec[5])
    hip0=3*minlw/16
    dhip=np.arange(-minlw/16,minlw/16,max(2,minlw/32))
    dhip0=np.arange(-1,1,0.4)
    if len(dhip)<=1:
        dhip=np.arange(-1,1,1)

    dzeave_m=np.arange(-0.5,0.5,0.3)
    dzridge_m=np.arange(-0.5,0.5,0.3)
    zeave_m=para3d_g[mg_idx][0]
    zridge_m=para3d_g[mg_idx][1]
    para3d_m0=[]
    para3d_m=[]
    for hip in hip0+dhip:
        para3d_temp=calculate_mansard_height(rec,hip,zeave_m,zridge_m,dzeave_m,dzridge_m,mask_ht,mask_rec,bmask)
        para3d_m0.extend(para3d_temp)
    mm_idx0=np.argmin(np.array(para3d_m0), axis=0)[5]
    hip0=para3d_m0[mm_idx0][2]
    for hip in hip0+dhip0:
        para3d_temp=calculate_mansard_height(rec,hip,zeave_m,zridge_m,dzeave_m,dzridge_m,mask_ht,mask_rec,bmask)
        para3d_m.extend(para3d_temp)
    mm_idx=np.argmin(np.array(para3d_m), axis=0)[5]
    return para3d_m,mm_idx

def calculate_mansard_height(rec,hip,hight0,zridge_m,dzeave_m,dzridge_m,mask_ht,mask_rec,bmask):
    para3d_temp=[]
    (msize,nsize)=np.shape(bmask)
    rt1, rt2=hip/rec[4],hip/rec[5]
    gxy=np.array([rec[8]*(1-rt1)*(1-rt2)+rec[10]*rt1*(1-rt2)+rec[12]*(1-rt1)*rt2+rec[14]*rt1*rt2,\
            rec[9]*(1-rt1)*(1-rt2)+rec[11]*rt1*(1-rt2)+rec[13]*(1-rt1)*rt2+rec[15]*rt1*rt2,\
            rec[8]*rt1*(1-rt2)+rec[10]*(1-rt1)*(1-rt2)+rec[12]*rt1*rt2+rec[14]*(1-rt1)*rt2,\
            rec[9]*rt1*(1-rt2)+rec[11]*(1-rt1)*(1-rt2)+rec[13]*rt1*rt2+rec[15]*(1-rt1)*rt2,\
            rec[8]*(1-rt1)*rt2+rec[10]*rt1*rt2+rec[12]*(1-rt1)*(1-rt2)+rec[14]*rt1*(1-rt2),\
            rec[9]*(1-rt1)*rt2+rec[11]*rt1*rt2+rec[13]*(1-rt1)*(1-rt2)+rec[15]*rt1*(1-rt2),\
            rec[8]*rt1*rt2+rec[10]*(1-rt1)*rt2+rec[12]*rt1*(1-rt2)+rec[14]*(1-rt1)*(1-rt2),\
            rec[9]*rt1*rt2+rec[11]*(1-rt1)*rt2+rec[13]*rt1*(1-rt2)+rec[15]*(1-rt1)*(1-rt2)])
    ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]],[gxy[2],gxy[3]]])
    ptn2=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]],[gxy[4],gxy[5]]])
    ptn3=np.array([[rec[12],rec[13]],[gxy[4],gxy[5]],[rec[14],rec[15]],[gxy[6],gxy[7]]])
    ptn4=np.array([[rec[10],rec[11]],[gxy[2],gxy[3]],[rec[14],rec[15]],[gxy[6],gxy[7]]])
    ptn5=np.array([[gxy[0],gxy[1]],[gxy[2],gxy[3]],[gxy[4],gxy[5]],[gxy[6],gxy[7]]])
    lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
    mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
    lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
    mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
    lsq3=np.hstack((ptn3,np.ones((4,1),dtype=float)))
    mask3=fitrec(msize,nsize,ptn3[:,0],ptn3[:,1])
    lsq4=np.hstack((ptn4,np.ones((4,1),dtype=float)))
    mask4=fitrec(msize,nsize,ptn4[:,0],ptn4[:,1])
    mask5=fitrec(msize,nsize,ptn5[:,0],ptn5[:,1])

    for zeave in hight0+dzeave_m:
        for zridge in zridge_m+dzridge_m:
            if zridge-zeave<0.5:
                continue
            ptz=np.array([zeave,zridge,zeave,zridge]).T
            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz,lsq1,mask1)
            maskz=fitmod(maskz,ptz,lsq2,mask2)
            maskz=fitmod(maskz,ptz,lsq3,mask3)
            maskz=fitmod(maskz,ptz,lsq4,mask4)
            maskz[np.where(mask5==1)]=zridge
            mask_diff=mask_ht-mask_rec*maskz
            mask_diff[np.where(np.absolute(mask_diff)>10)]=0
            rmse_m=np.sum(((mask_diff)**2)*bmask)/np.sum(mask_rec*bmask)
            para3d_temp.append(np.array([zeave,zridge,hip,hip,1,rmse_m]))
    return para3d_temp

def dsm_fitting(shape2d,imgsize_t,img_dsm_t,L_mask,para3d_type):    
    para3d=[]
    build_dsm=copy.deepcopy(img_dsm_t)
    
    for i, rec in enumerate(shape2d):
        roof_type=np.argmin(para3d_type[i][5::6])
        para3d_temp=np.hstack((roof_type+1,para3d_type[i][roof_type*6:roof_type*6+6]))
        
        (msize,nsize)=np.shape(build_dsm)
        tx, ty=np.rint(rec[8::2]),np.rint(rec[9::2])
        mask_rec=fitrec(msize,nsize,tx,ty)
        mask_ht=mask_rec*build_dsm
        if para3d_temp[0]==1:
            build_dsm[np.where(mask_rec==1)]=para3d_temp[1]
        elif para3d_temp[0]==2:
            gxyp=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
            gxyv=np.array([(rec[8]+rec[12])/2,(rec[9]+rec[13])/2,(rec[10]+rec[14])/2,(rec[11]+rec[15])/2])
            ptz=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1],para3d_temp[2]]).T
            if para3d_temp[5]==1:
                ptnp1=np.array([[rec[8],rec[9]],[gxyp[0],gxyp[1]],[rec[12],rec[13]],[gxyp[2],gxyp[3]]])
                lsqp1=np.hstack((ptnp1,np.ones((4,1),dtype=float)))
                maskp1=fitrec(msize,nsize,ptnp1[:,0],ptnp1[:,1])
                ptnp2=np.array([[rec[10],rec[11]],[gxyp[0],gxyp[1]],[rec[14],rec[15]],[gxyp[2],gxyp[3]]])
                lsqp2=np.hstack((ptnp2,np.ones((4,1),dtype=float)))
                maskp2=mask_rec-maskp1
                maskpz=np.zeros((msize,nsize),dtype=float)
                maskpz=fitmod(maskpz,ptz,lsqp1,maskp1)
                maskpz=fitmod(maskpz,ptz,lsqp2,maskp2)
                build_dsm[np.where(mask_rec==1)]=maskpz[np.where(mask_rec==1)]
            else:
                ptnv1=np.array([[rec[8],rec[9]],[gxyv[0],gxyv[1]],[rec[10],rec[11]],[gxyv[2],gxyv[3]]])
                lsqv1=np.hstack((ptnv1,np.ones((4,1),dtype=float)))
                maskv1=fitrec(msize,nsize,ptnv1[:,0],ptnv1[:,1])
                ptnv2=np.array([[rec[12],rec[13]],[gxyv[0],gxyv[1]],[rec[14],rec[15]],[gxyv[2],gxyv[3]]])
                lsqv2=np.hstack((ptnv2,np.ones((4,1),dtype=float)))
                maskv2=mask_rec-maskv1
                maskvz=np.zeros((msize,nsize),dtype=float)
                maskvz=fitmod(maskvz,ptz,lsqv1,maskv1)
                maskvz=fitmod(maskvz,ptz,lsqv2,maskv2)
                build_dsm[np.where(mask_rec==1)]=maskvz[np.where(mask_rec==1)]
        elif para3d_temp[0]==3:
            ptz=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1],para3d_temp[2]]).T
            ptz_tr=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1]]).T
            if para3d_temp[5]==1:
                rt=para3d_temp[3]/rec[4]
                gxy0=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
                gxy=np.array([gxy0[0]*(1-rt)+gxy0[2]*rt,gxy0[1]*(1-rt)+gxy0[3]*rt,\
                              gxy0[0]*rt+gxy0[2]*(1-rt),gxy0[1]*rt+gxy0[3]*(1-rt)])
                ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]],[gxy[2],gxy[3]]])
                ptn2=np.array([[rec[10],rec[11]],[gxy[0],gxy[1]],[rec[14],rec[15]],[gxy[2],gxy[3]]])
                ptn3=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]]])
                ptn4=np.array([[rec[12],rec[13]],[gxy[2],gxy[3]],[rec[14],rec[15]]])

                lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
                mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
                lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
                mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
                lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
                mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
                lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
                mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])
                maskz=np.zeros((msize,nsize),dtype=float)
                maskz=fitmod(maskz,ptz,lsq1,mask1)
                maskz=fitmod(maskz,ptz,lsq2,mask2)
                maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
                maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
                build_dsm[np.where(mask_rec==1)]=maskz[np.where(mask_rec==1)]
            else:
                rt=para3d_temp[3]/rec[5]
                gxy0=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
                gxy=np.array([gxy0[0]*(1-rt)+gxy0[2]*rt,gxy0[1]*(1-rt)+gxy0[3]*rt,\
                              gxy0[0]*rt+gxy0[2]*(1-rt),gxy0[1]*rt+gxy0[3]*(1-rt)])
                ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]],[gxy[2],gxy[3]]])
                ptn2=np.array([[rec[12],rec[13]],[gxy[0],gxy[1]],[rec[14],rec[15]],[gxy[2],gxy[3]]])
                ptn3=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]]])
                ptn4=np.array([[rec[10],rec[11]],[gxy[2],gxy[3]],[rec[14],rec[15]]])

                lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
                mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
                lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
                mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
                lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
                mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
                lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
                mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])
                maskz=np.zeros((msize,nsize),dtype=float)
                maskz=fitmod(maskz,ptz,lsq1,mask1)
                maskz=fitmod(maskz,ptz,lsq2,mask2)
                maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
                maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
                build_dsm[np.where(mask_rec==1)]=maskz[np.where(mask_rec==1)]
        elif para3d_temp[0]==4:
            ptz_tr=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1]]).T
            gxy=np.array([np.mean(rec[8::2]),np.mean(rec[9::2])])
            ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]]])
            ptn2=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]]])
            ptn3=np.array([[rec[10],rec[11]],[gxy[0],gxy[1]],[rec[14],rec[15]]])
            ptn4=np.array([[rec[12],rec[13]],[gxy[0],gxy[1]],[rec[14],rec[15]]])
            lsq1=np.hstack((ptn1,np.ones((3,1),dtype=float)))
            mask1=fitrec_tr(msize,nsize,ptn3[:,0],ptn1[:,1])
            lsq2=np.hstack((ptn2,np.ones((3,1),dtype=float)))
            mask2=fitrec_tr(msize,nsize,ptn3[:,0],ptn2[:,1])
            lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
            mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
            lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
            mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])

            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz_tr,lsq1,mask1)
            maskz=fitmod(maskz,ptz_tr,lsq2,mask2)
            maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
            maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
            build_dsm[np.where(mask_rec==1)]=maskz[np.where(mask_rec==1)]
        elif para3d_temp[0]==5:
            ptz=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1],para3d_temp[2]]).T
            rt1=para3d_temp[3]/rec[4]
            rt2=para3d_temp[3]/rec[5]
            gxy=np.array([rec[8]*(1-rt1)*(1-rt2)+rec[10]*rt1*(1-rt2)+rec[12]*(1-rt1)*rt2+rec[14]*rt1*rt2,\
                rec[9]*(1-rt1)*(1-rt2)+rec[11]*rt1*(1-rt2)+rec[13]*(1-rt1)*rt2+rec[15]*rt1*rt2,\
                rec[8]*rt1*(1-rt2)+rec[10]*(1-rt1)*(1-rt2)+rec[12]*rt1*rt2+rec[14]*(1-rt1)*rt2,\
                rec[9]*rt1*(1-rt2)+rec[11]*(1-rt1)*(1-rt2)+rec[13]*rt1*rt2+rec[15]*(1-rt1)*rt2,\
                rec[8]*(1-rt1)*rt2+rec[10]*rt1*rt2+rec[12]*(1-rt1)*(1-rt2)+rec[14]*rt1*(1-rt2),\
                rec[9]*(1-rt1)*rt2+rec[11]*rt1*rt2+rec[13]*(1-rt1)*(1-rt2)+rec[15]*rt1*(1-rt2),\
                rec[8]*rt1*rt2+rec[10]*(1-rt1)*rt2+rec[12]*rt1*(1-rt2)+rec[14]*(1-rt1)*(1-rt2),\
                rec[9]*rt1*rt2+rec[11]*(1-rt1)*rt2+rec[13]*rt1*(1-rt2)+rec[15]*(1-rt1)*(1-rt2)])
            ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]],[gxy[2],gxy[3]]])
            ptn2=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]],[gxy[4],gxy[5]]])
            ptn3=np.array([[rec[12],rec[13]],[gxy[4],gxy[5]],[rec[14],rec[15]],[gxy[6],gxy[7]]])
            ptn4=np.array([[rec[10],rec[11]],[gxy[2],gxy[3]],[rec[14],rec[15]],[gxy[6],gxy[7]]])
            ptn5=np.array([[gxy[0],gxy[1]],[gxy[2],gxy[3]],[gxy[4],gxy[5]],[gxy[6],gxy[7]]])
            lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
            mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
            lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
            mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
            lsq3=np.hstack((ptn3,np.ones((4,1),dtype=float)))
            mask3=fitrec(msize,nsize,ptn3[:,0],ptn3[:,1])
            lsq4=np.hstack((ptn4,np.ones((4,1),dtype=float)))
            mask4=fitrec(msize,nsize,ptn4[:,0],ptn4[:,1])
            mask5=fitrec(msize,nsize,ptn5[:,0],ptn5[:,1])
            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz,lsq1,mask1)
            maskz=fitmod(maskz,ptz,lsq2,mask2)
            maskz=fitmod(maskz,ptz,lsq3,mask3)
            maskz=fitmod(maskz,ptz,lsq4,mask4)
            maskz[np.where(mask5==1)]=ptz[1]
            build_dsm[np.where(mask_rec==1)]=maskz[np.where(mask_rec==1)]
        para3d.append(para3d_temp)
    
    return build_dsm

def dsm_fitting_fast(shape3d,img_dsm_t):
    shape2d=shape3d[0:16]
    para3d_type=shape3d[16:]
    rec = copy.deepcopy(shape2d)

    para3d_temp = para3d_type

    # generate mask in local area
    range_local = np.ceil(np.array([min(rec[9::2]) - 10, max(rec[9::2]) + 10, \
                                    min(rec[8::2]) - 10, max(rec[8::2]) + 10])).astype(np.int)
    dsm_loc = copy.deepcopy(img_dsm_t[range_local[0]:range_local[1], range_local[2]:range_local[3]])

    rec[8::2] -= range_local[2]
    rec[9::2] -= range_local[0]
    tx, ty = np.rint(rec[8::2]), np.rint(rec[9::2])
    (msize, nsize) = np.shape(dsm_loc)
    tx[np.where(tx > nsize - 1)] = nsize - 1
    ty[np.where(ty > msize - 1)] = msize - 1

    mask_rec = fitrec(msize, nsize, tx, ty)
    mask_ht = mask_rec * dsm_loc
    if para3d_temp[0] == 1:
        mask_ht[np.where(mask_rec == 1)] = para3d_temp[1]
    elif para3d_temp[0] == 2:
        gxyp=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
        gxyv=np.array([(rec[8]+rec[12])/2,(rec[9]+rec[13])/2,(rec[10]+rec[14])/2,(rec[11]+rec[15])/2])
        ptz=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1],para3d_temp[2]]).T
        if para3d_temp[5]==1:
            ptnp1=np.array([[rec[8],rec[9]],[gxyp[0],gxyp[1]],[rec[12],rec[13]],[gxyp[2],gxyp[3]]])
            lsqp1=np.hstack((ptnp1,np.ones((4,1),dtype=float)))
            maskp1=fitrec(msize,nsize,ptnp1[:,0],ptnp1[:,1])
            ptnp2=np.array([[rec[10],rec[11]],[gxyp[0],gxyp[1]],[rec[14],rec[15]],[gxyp[2],gxyp[3]]])
            lsqp2=np.hstack((ptnp2,np.ones((4,1),dtype=float)))
            maskp2=mask_rec-maskp1
            maskpz=np.zeros((msize,nsize),dtype=float)
            maskpz=fitmod(maskpz,ptz,lsqp1,maskp1)
            maskpz=fitmod(maskpz,ptz,lsqp2,maskp2)
            mask_ht[np.where(mask_rec == 1)] = maskpz[np.where(mask_rec == 1)]
        else:
            ptnv1=np.array([[rec[8],rec[9]],[gxyv[0],gxyv[1]],[rec[10],rec[11]],[gxyv[2],gxyv[3]]])
            lsqv1=np.hstack((ptnv1,np.ones((4,1),dtype=float)))
            maskv1=fitrec(msize,nsize,ptnv1[:,0],ptnv1[:,1])
            ptnv2=np.array([[rec[12],rec[13]],[gxyv[0],gxyv[1]],[rec[14],rec[15]],[gxyv[2],gxyv[3]]])
            lsqv2=np.hstack((ptnv2,np.ones((4,1),dtype=float)))
            maskv2=mask_rec-maskv1
            maskvz=np.zeros((msize,nsize),dtype=float)
            maskvz=fitmod(maskvz,ptz,lsqv1,maskv1)
            maskvz=fitmod(maskvz,ptz,lsqv2,maskv2)
            mask_ht[np.where(mask_rec == 1)] = maskvz[np.where(mask_rec == 1)]
    elif para3d_temp[0]==3:
        ptz=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1],para3d_temp[2]]).T
        ptz_tr=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1]]).T
        if para3d_temp[5]==1:
            rt=para3d_temp[3]/rec[4]
            gxy0=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
            gxy=np.array([gxy0[0]*(1-rt)+gxy0[2]*rt,gxy0[1]*(1-rt)+gxy0[3]*rt,\
                          gxy0[0]*rt+gxy0[2]*(1-rt),gxy0[1]*rt+gxy0[3]*(1-rt)])
            ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]],[gxy[2],gxy[3]]])
            ptn2=np.array([[rec[10],rec[11]],[gxy[0],gxy[1]],[rec[14],rec[15]],[gxy[2],gxy[3]]])
            ptn3=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]]])
            ptn4=np.array([[rec[12],rec[13]],[gxy[2],gxy[3]],[rec[14],rec[15]]])
            lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
            mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
            lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
            mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
            lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
            mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
            lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
            mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])
            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz,lsq1,mask1)
            maskz=fitmod(maskz,ptz,lsq2,mask2)
            maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
            maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
            mask_ht[np.where(mask_rec == 1)] = maskz[np.where(mask_rec == 1)]
        else:
            rt=para3d_temp[3]/rec[5]
            gxy0=np.array([(rec[8]+rec[10])/2,(rec[9]+rec[11])/2,(rec[12]+rec[14])/2,(rec[13]+rec[15])/2])
            gxy=np.array([gxy0[0]*(1-rt)+gxy0[2]*rt,gxy0[1]*(1-rt)+gxy0[3]*rt,\
                          gxy0[0]*rt+gxy0[2]*(1-rt),gxy0[1]*rt+gxy0[3]*(1-rt)])
            ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]],[gxy[2],gxy[3]]])
            ptn2=np.array([[rec[12],rec[13]],[gxy[0],gxy[1]],[rec[14],rec[15]],[gxy[2],gxy[3]]])
            ptn3=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]]])
            ptn4=np.array([[rec[10],rec[11]],[gxy[2],gxy[3]],[rec[14],rec[15]]])

            lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
            mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
            lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
            mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
            lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
            mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
            lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
            mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])
            maskz=np.zeros((msize,nsize),dtype=float)
            maskz=fitmod(maskz,ptz,lsq1,mask1)
            maskz=fitmod(maskz,ptz,lsq2,mask2)
            maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
            maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
            mask_ht[np.where(mask_rec == 1)] = maskz[np.where(mask_rec == 1)]
    elif para3d_temp[0]==4:
        ptz_tr=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1]]).T
        gxy=np.array([np.mean(rec[8::2]),np.mean(rec[9::2])])
        ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]]])
        ptn2=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]]])
        ptn3=np.array([[rec[10],rec[11]],[gxy[0],gxy[1]],[rec[14],rec[15]]])
        ptn4=np.array([[rec[12],rec[13]],[gxy[0],gxy[1]],[rec[14],rec[15]]])
        lsq1=np.hstack((ptn1,np.ones((3,1),dtype=float)))
        mask1=fitrec_tr(msize,nsize,ptn3[:,0],ptn1[:,1])
        lsq2=np.hstack((ptn2,np.ones((3,1),dtype=float)))
        mask2=fitrec_tr(msize,nsize,ptn3[:,0],ptn2[:,1])
        lsq3=np.hstack((ptn3,np.ones((3,1),dtype=float)))
        mask3=fitrec_tr(msize,nsize,ptn3[:,0],ptn3[:,1])
        lsq4=np.hstack((ptn4,np.ones((3,1),dtype=float)))
        mask4=fitrec_tr(msize,nsize,ptn4[:,0],ptn4[:,1])

        maskz=np.zeros((msize,nsize),dtype=float)
        maskz=fitmod(maskz,ptz_tr,lsq1,mask1)
        maskz=fitmod(maskz,ptz_tr,lsq2,mask2)
        maskz=fitmod(maskz,ptz_tr,lsq3,mask3)
        maskz=fitmod(maskz,ptz_tr,lsq4,mask4)
        mask_ht[np.where(mask_rec == 1)] = maskz[np.where(mask_rec == 1)]
    elif para3d_temp[0]==5:
        ptz=np.array([para3d_temp[1],para3d_temp[2],para3d_temp[1],para3d_temp[2]]).T
        rt1=para3d_temp[3]/rec[4]
        rt2=para3d_temp[3]/rec[5]
        gxy=np.array([rec[8]*(1-rt1)*(1-rt2)+rec[10]*rt1*(1-rt2)+rec[12]*(1-rt1)*rt2+rec[14]*rt1*rt2,\
            rec[9]*(1-rt1)*(1-rt2)+rec[11]*rt1*(1-rt2)+rec[13]*(1-rt1)*rt2+rec[15]*rt1*rt2,\
            rec[8]*rt1*(1-rt2)+rec[10]*(1-rt1)*(1-rt2)+rec[12]*rt1*rt2+rec[14]*(1-rt1)*rt2,\
            rec[9]*rt1*(1-rt2)+rec[11]*(1-rt1)*(1-rt2)+rec[13]*rt1*rt2+rec[15]*(1-rt1)*rt2,\
            rec[8]*(1-rt1)*rt2+rec[10]*rt1*rt2+rec[12]*(1-rt1)*(1-rt2)+rec[14]*rt1*(1-rt2),\
            rec[9]*(1-rt1)*rt2+rec[11]*rt1*rt2+rec[13]*(1-rt1)*(1-rt2)+rec[15]*rt1*(1-rt2),\
            rec[8]*rt1*rt2+rec[10]*(1-rt1)*rt2+rec[12]*rt1*(1-rt2)+rec[14]*(1-rt1)*(1-rt2),\
            rec[9]*rt1*rt2+rec[11]*(1-rt1)*rt2+rec[13]*rt1*(1-rt2)+rec[15]*(1-rt1)*(1-rt2)])
        ptn1=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[10],rec[11]],[gxy[2],gxy[3]]])
        ptn2=np.array([[rec[8],rec[9]],[gxy[0],gxy[1]],[rec[12],rec[13]],[gxy[4],gxy[5]]])
        ptn3=np.array([[rec[12],rec[13]],[gxy[4],gxy[5]],[rec[14],rec[15]],[gxy[6],gxy[7]]])
        ptn4=np.array([[rec[10],rec[11]],[gxy[2],gxy[3]],[rec[14],rec[15]],[gxy[6],gxy[7]]])
        ptn5=np.array([[gxy[0],gxy[1]],[gxy[2],gxy[3]],[gxy[4],gxy[5]],[gxy[6],gxy[7]]])
        lsq1=np.hstack((ptn1,np.ones((4,1),dtype=float)))
        mask1=fitrec(msize,nsize,ptn1[:,0],ptn1[:,1])
        lsq2=np.hstack((ptn2,np.ones((4,1),dtype=float)))
        mask2=fitrec(msize,nsize,ptn2[:,0],ptn2[:,1])
        lsq3=np.hstack((ptn3,np.ones((4,1),dtype=float)))
        mask3=fitrec(msize,nsize,ptn3[:,0],ptn3[:,1])
        lsq4=np.hstack((ptn4,np.ones((4,1),dtype=float)))
        mask4=fitrec(msize,nsize,ptn4[:,0],ptn4[:,1])
        mask5=fitrec(msize,nsize,ptn5[:,0],ptn5[:,1])
        maskz=np.zeros((msize,nsize),dtype=float)
        maskz=fitmod(maskz,ptz,lsq1,mask1)
        maskz=fitmod(maskz,ptz,lsq2,mask2)
        maskz=fitmod(maskz,ptz,lsq3,mask3)
        maskz=fitmod(maskz,ptz,lsq4,mask4)
        maskz[np.where(mask5 == 1)] = ptz[1]
        mask_ht[np.where(mask_rec == 1)] = maskz[np.where(mask_rec == 1)]
    dsmlist=dsm2list(mask_ht, range_local)

    return dsmlist

def dsm2list(mask_ht, range_local):
    (recr,recc)=np.where(mask_ht!=0)
    dsmlist=np.vstack((recr,recc,mask_ht[recr,recc])).T
    dsmlist[:, 0] += range_local[0]
    dsmlist[:, 1] += range_local[2]

    return dsmlist

def list2dsm(dsmlist,img_dsm_t):
    building_dsm = copy.deepcopy(img_dsm_t)
    for i,dlist in enumerate(dsmlist):
        pt_build=np.array(dlist)
        pt_loc=pt_build[:,0:2].astype(int)
        building_dsm[pt_loc[:,0],pt_loc[:,1]]=pt_build[:,2]
    return building_dsm

def mutli_fit(shape2d,imgsize_t,L_mask,img_dsm_t):
    rec=shape2d
    img_mask=np.zeros((imgsize_t[0],imgsize_t[1]),dtype=np.uint8)
    img_mask[np.where(L_mask==rec[0]+1)]=1
    
    #generate mask in local area
    range_local=np.ceil(np.array([min(rec[9::2])-10,max(rec[9::2])+10,min(rec[8::2])-10,max(rec[8::2])+10])).astype(np.int)
    dsm_loc=copy.deepcopy(img_dsm_t[range_local[0]:range_local[1],range_local[2]:range_local[3]])
    
    rec[8::2]-=range_local[2]
    rec[9::2]-=range_local[0]
    tx, ty=np.rint(rec[8::2]),np.rint(rec[9::2])
    (msize, nsize)=np.shape(dsm_loc)
    tx[np.where(tx>nsize-1)]=nsize-1
    ty[np.where(ty>msize-1)]=msize-1
    
    mask_rec=fitrec(msize,nsize,tx,ty)
    mask_ht=mask_rec*dsm_loc
    bmask=copy.deepcopy(img_mask[range_local[0]:range_local[1],range_local[2]:range_local[3]])
    hight0=np.sum(mask_ht*bmask)/np.sum(mask_rec*bmask)
    dzeave=np.arange(-3,3,0.3)
    dzridge=np.arange(0.5,4,0.3)
    del_ht=0.5
    
    #1.flat type
    para3d_f,mf_idx=fit_flat(hight0,dzeave,mask_ht,mask_rec,bmask)
    
    #2.gable type
    para3d_g,mg_idx=fit_gable(hight0,dzeave,dzridge,del_ht,rec,mask_ht,mask_rec,bmask)
    
    #3.hip
    para3d_h,mh_idx=fit_hip(hight0,para3d_g,mg_idx,rec,mask_ht,mask_rec,bmask)
    
    #4.pyramid
    para3d_p,mp_idx=fit_gable(hight0,dzeave,dzridge,del_ht,rec,mask_ht,mask_rec,bmask)
    
    #5.mansard
    para3d_m,mm_idx=fit_mansard(hight0,para3d_g,mg_idx,rec,mask_ht,mask_rec,bmask)
    
    return np.array([para3d_f[mf_idx],para3d_g[mg_idx],para3d_h[mh_idx],para3d_p[mp_idx],para3d_m[mm_idx]])


def single_fit(shape2d, imgsize_t, L_mask, img_dsm_t):
    rec = shape2d
    img_mask = np.zeros((imgsize_t[0], imgsize_t[1]), dtype=np.uint8)
    img_mask[np.where(L_mask == rec[0] + 1)] = 1

    # generate mask in local area
    range_local = np.ceil(
        np.array([min(rec[9::2]) - 10, max(rec[9::2]) + 10, min(rec[8::2]) - 10, max(rec[8::2]) + 10])).astype(np.int)
    dsm_loc = copy.deepcopy(img_dsm_t[range_local[0]:range_local[1], range_local[2]:range_local[3]])

    rec[8::2] -= range_local[2]
    rec[9::2] -= range_local[0]
    tx, ty = np.rint(rec[8::2]), np.rint(rec[9::2])
    (msize, nsize) = np.shape(dsm_loc)
    tx[np.where(tx > nsize - 1)] = nsize - 1
    ty[np.where(ty > msize - 1)] = msize - 1

    mask_rec = fitrec(msize, nsize, tx, ty)
    mask_ht = mask_rec * dsm_loc
    bmask = copy.deepcopy(img_mask[range_local[0]:range_local[1], range_local[2]:range_local[3]])
    hight0 = np.sum(mask_ht * bmask) / np.sum(mask_rec * bmask)
    dzeave = np.arange(-3, 3, 0.3)
    dzridge = np.arange(0.5, 4, 0.3)
    del_ht = 0.5

    # 1.flat type
    para3d_f, mf_idx = fit_flat(hight0, dzeave, mask_ht, mask_rec, bmask)

    # 2.gable type
    para3d_g, mg_idx = fit_gable(hight0, dzeave, dzridge, del_ht, rec, mask_ht, mask_rec, bmask)

    # 3.hip
    para3d_h, mh_idx = fit_hip(hight0, para3d_g, mg_idx, rec, mask_ht, mask_rec, bmask)

    # 4.pyramid
    para3d_p, mp_idx = fit_gable(hight0, dzeave, dzridge, del_ht, rec, mask_ht, mask_rec, bmask)

    # 5.mansard
    para3d_m, mm_idx = fit_mansard(hight0, para3d_g, mg_idx, rec, mask_ht, mask_rec, bmask)

    return np.array([para3d_f[mf_idx], para3d_g[mg_idx], para3d_h[mh_idx], para3d_p[mp_idx], para3d_m[mm_idx]])








    