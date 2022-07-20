from skimage import io,data,filters,segmentation,measure,morphology,color,feature,draw
from skimage.util import img_as_ubyte
from scipy import ndimage,optimize,signal
import numpy as np
from math import pi,sqrt,sin,cos,ceil,floor,degrees
import cv2
import json
import copy

import pdb 

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  (h,w)=image.shape[:2]
  rot_mat = cv2.getRotationMatrix2D(image_center, degrees(angle), 1.0)
  nw=int(h*np.abs(rot_mat[0,1])+w*np.abs(rot_mat[0,0]))
  nh=int(h*np.abs(rot_mat[0,0])+w*np.abs(rot_mat[0,1]))
  rot_mat[0,2]+=(nw/2)-image_center[0]
  rot_mat[1,2]+=(nh/2)-image_center[1]
  if h*w<1:
      result_img=image
  else:
      result_img = cv2.warpAffine(image, rot_mat, (nw,nh), flags=cv2.INTER_LINEAR)
  return result_img

####################################################
# maximum inner rectangle detection part is from https://github.com/pogam/ExtractRect
def findMaxRect(data):
    '''http://stackoverflow.com/a/30418912/5008845'''
    nrows,ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    skip = 1
    area_max = (0, [])
   
    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh+1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, c-minw+1, r, c)])

    return area_max

def residual(angle,data):
    nx,ny = data.shape
    M = cv2.getRotationMatrix2D(((nx-1)/2,(ny-1)/2),angle,1)
    RotData = cv2.warpAffine(data,M,(nx,ny),flags=cv2.INTER_NEAREST,borderValue=1)
    rectangle = findMaxRect(RotData)
   
    return 1./rectangle[0]

def residual_star(args):
    return residual(*args)
    
def get_rectangle_coord(angle,data,flag_out=None):
    nx,ny = data.shape
    M = cv2.getRotationMatrix2D(((nx-1)/2,(ny-1)/2),angle,1)
    RotData = cv2.warpAffine(data,M,(nx,ny),flags=cv2.INTER_NEAREST,borderValue=1)
    rectangle = findMaxRect(RotData)    
    if flag_out:
        return rectangle[1][0], M, RotData
    else:
        return rectangle[1][0], M

def findRotMaxRect(data_in,flag_opt=False,flag_parallel = False, main_angle=0,flag_out=None,flag_enlarge_img=False,limit_image_size=300):
    '''
    flag_opt     : True only nbre_angle are tested between 90 and 180 
                        and a opt descent algo is run on the best fit
                   False 100 angle are tested from 90 to 180.
    flag_parallel: only valid when flag_opt=False. the 100 angle are run on multithreading
    flag_out     : angle and rectangle of the rotated image are output together with the rectangle of the original image
    flag_enlarge_img : the image used in the function is double of the size of the original to ensure all feature stay in when rotated
    limit_image_size : control the size numbre of pixel of the image use in the function. 
                       this speeds up the code but can give approximated results if the shape is not simple
    '''
    nx_in, ny_in = data_in.shape
    if nx_in != ny_in:
        n = max([nx_in,ny_in])
        data_square = np.ones([n,n])
        xshift = round((n-nx_in)/2)
        yshift = round((n-ny_in)/2)

        if yshift == 0 and n-ny_in<=0:
            data_square[xshift:(xshift+nx_in),:                 ] = data_in[:,:]
        else: 
            data_square[:                 ,yshift:(yshift+ny_in)] = data_in[:,:]
    else:
        xshift = 0
        yshift = 0
        data_square = data_in

    #apply scale factor if image bigger than limit_image_size
    if data_square.shape[0] > limit_image_size:
        data_small = cv2.resize(data_square,(limit_image_size, limit_image_size),interpolation=0)
        scale_factor = 1.*data_square.shape[0]/data_small.shape[0]
    else:
        data_small = data_square
        scale_factor = 1


    # set the input data with an odd number of point in each dimension to make rotation easier
    nx,ny = data_small.shape
    nx_extra = -nx; ny_extra = -ny   
    if nx%2==0:
        nx+=1
        nx_extra = 1
    if ny%2==0:
        ny+=1
        ny_extra = 1
    data_odd = np.ones([data_small.shape[0]+max([0,nx_extra]),data_small.shape[1]+max([0,ny_extra])])
    data_odd[:-nx_extra, :-ny_extra] = data_small
    nx,ny = data_odd.shape

    nx_odd,ny_odd = data_odd.shape
    if flag_enlarge_img:
        data = np.zeros([2*data_odd.shape[0]+1,2*data_odd.shape[1]+1]) + 1
        nx,ny = data.shape
        data[nx/2-nx_odd/2:nx/2+nx_odd/2,ny/2-ny_odd/2:ny/2+ny_odd/2] = data_odd
    else:
        data = np.copy(data_odd)
        nx,ny = data.shape

    if main_angle<0:
        main_angle+=pi/2
    angle_selected = main_angle*180/pi
    rectangle, M_rect_max, RotData  = get_rectangle_coord(angle_selected,data,flag_out=True)

    #invert rectangle 
    M_invert = cv2.invertAffineTransform(M_rect_max)
    rect_coord = [rectangle[:2], [rectangle[0],rectangle[3]] , 
                  rectangle[2:], [rectangle[2],rectangle[1]] ]

    rect_coord_ori = []
    for coord in rect_coord:
        rect_coord_ori.append(np.dot(M_invert,[coord[0],(ny-1)-coord[1],1]))

    #transform to numpy coord of input image
    coord_out = []
    for coord in rect_coord_ori:
        coord_out.append(    [ scale_factor*round(       coord[0]-(nx/2-nx_odd/2),0)-xshift,\
                               scale_factor*round((ny-1)-coord[1]-(ny/2-ny_odd/2),0)-yshift])
    
    coord_out_rot = []
    coord_out_rot_h = []
    for coord in rect_coord:
        coord_out_rot.append( [ scale_factor*round(       coord[0]-(nx/2-nx_odd/2),0)-xshift, \
                                scale_factor*round(       coord[1]-(ny/2-ny_odd/2),0)-yshift ])
        coord_out_rot_h.append( [ scale_factor*round(       coord[0]-(nx/2-nx_odd/2),0), \
                                  scale_factor*round(       coord[1]-(ny/2-ny_odd/2),0) ])

    if flag_out is None:
        return coord_out
    elif flag_out == 'rotation':
        return coord_out, angle_selected, coord_out_rot
    else:
        pdb.set_trace()


#########################################################
#get mask on next level of grid
def getgridmask(bd_imr,recori,im_0):
    (recr,recc)=np.where(bd_imr==1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bd_ime=im_0-cv2.dilate(bd_imr.astype(float),kernel,iterations = 1)
    (ms,ns)=np.shape(im_0)
    if len(recr)<1:
        decp=[]
        bd_imf=bd_imr
        return decp,bd_imf
    if len(recc)<1:
        decp=[]
        bd_imf=bd_imr
        return decp,bd_imf
    if recori==0:
        minr=min(recr)
        maxr=max(recr)
        minc=min(recc)
        maxc=max(recc)
        if maxc-minc < 2:
            maxc += 1
        if maxr-minr < 2:
            maxr += 1
        rec1=np.array([minr,maxr,minc,maxc])
        if bd_ime[minr:maxr,minc:maxc].min()==-1:
            if bd_ime[minr:maxr,minc+1:maxc].min()==0:
                rec1[2]=minc+1
            if bd_ime[minr:maxr,minc:maxc-1].min()==0:
                rec1[3]=maxc-1
            if bd_ime[minr+1:maxr,minc:maxc].min()==0:
                rec1[0]=minr+1
            if bd_ime[minr:maxr-1,minc:maxc].min()==0:
                rec1[1]=maxr-1
        if bd_ime[minr:maxr,minc-1].min()==0:
            rec1[2]=minc-1
        if bd_ime[minr:maxr,maxc+1].min()==0:
            rec1[3]=maxc+1
        if bd_ime[minr-1,minc:maxc].min()==0:
            rec1[0]=minr-1
        if bd_ime[maxr+1,minc:maxc].min()==0:
            rec1[1]=maxr+1
        
        if bd_ime[rec1[0]:rec1[1],rec1[2]:rec1[3]].min()==-1:
            LRout=findRotMaxRect((1-im_0), flag_opt=True, main_angle=0)
            for k, lrout in enumerate(LRout):
                lrout[0],lrout[1]=lrout[1],lrout[0]
            rec=np.array(LRout).T
            rec1=np.array([min(rec[1,:]),max(rec[1,:]),min(rec[0,:]),max(rec[0,:])])
        
        decp=np.array([[rec1[2],rec1[3],rec1[3],rec1[2]],[rec1[0],rec1[0],rec1[1],rec1[1]]])
        bd_imf=(0*bd_ime).astype(int)
        bd_imf[int(rec1[0]):int(rec1[1]),int(rec1[2]):int(rec1[3])]=1
    else:
        ori1=recori
        bd_imr_s=cv2.dilate(bd_imr.astype(float),kernel,iterations = 1)
        LRout_sd=findRotMaxRect((1-im_0), flag_opt=True, main_angle=ori1)
        for j, lrout in enumerate(LRout_sd):
            lrout[0],lrout[1]=lrout[1],lrout[0]
        rec=np.array(LRout_sd)
        dx11=rec[0,0]+sin(ori1)
        dy11=rec[0,1]-cos(ori1)
        dx21=rec[1,0]+sin(ori1)
        dy21=rec[1,1]-cos(ori1)
        dx22=rec[1,0]+cos(ori1)
        dy22=rec[1,1]+sin(ori1)
        dx32=rec[2,0]+cos(ori1)
        dy32=rec[2,1]+sin(ori1)
        dx33=rec[2,0]-sin(ori1)
        dy33=rec[2,1]+cos(ori1)
        dx43=rec[3,0]-sin(ori1)
        dy43=rec[3,1]+cos(ori1)
        dx44=rec[3,0]-cos(ori1)
        dy44=rec[3,1]-sin(ori1)
        dx14=rec[0,0]-cos(ori1)
        dy14=rec[0,1]-sin(ori1)
        bd_ime1=draw.polygon2mask([ms,ns],np.array([np.array([dy11,dy21,rec[2,1],rec[3,1]]).T,np.array([dx11,dx21,rec[2,0],rec[3,0]]).T]).T)-bd_imr_s
        bd_ime2=draw.polygon2mask([ms,ns],np.array([np.array([rec[0,1],dy22,dy32,rec[3,1]]).T,np.array([rec[0,0],dx22,dx32,rec[3,0]]).T]).T)-bd_imr_s
        bd_ime3=draw.polygon2mask([ms,ns],np.array([np.array([rec[0,1],rec[1,1],dy33,dy43]).T,np.array([rec[0,0],rec[1,0],dx33,dx43]).T]).T)-bd_imr_s
        bd_ime4=draw.polygon2mask([ms,ns],np.array([np.array([dy14,rec[1,1],rec[2,1],dy44]).T,np.array([dx14,rec[1,0],rec[2,0],dx44]).T]).T)-bd_imr_s
        ex1, ex2, ex3, ex4 = 0, 0, 0, 0
        if len(np.where(bd_ime==-1))>len(np.where(bd_ime1==-1)):
            ex1=1
        if len(np.where(bd_ime==-1))>len(np.where(bd_ime2==-1)):
            ex2=1
        if len(np.where(bd_ime==-1))>len(np.where(bd_ime3==-1)):
            ex3=1
        if len(np.where(bd_ime==-1))>len(np.where(bd_ime4==-1)):
            ex4=1
        dx1=rec[0,0]+sin(ori1)*ex1-cos(ori1)*ex4
        dy1=rec[0,1]-cos(ori1)*ex1-sin(ori1)*ex4
        dx2=rec[1,0]+sin(ori1)*ex1+cos(ori1)*ex2
        dy2=rec[1,1]-cos(ori1)*ex1+sin(ori1)*ex2
        dx3=rec[2,0]+cos(ori1)*ex2-sin(ori1)*ex3
        dy3=rec[2,1]+sin(ori1)*ex2+cos(ori1)*ex3
        dx4=rec[3,0]-sin(ori1)*ex3-cos(ori1)*ex4
        dy4=rec[3,1]+cos(ori1)*ex3-sin(ori1)*ex4
        decp=np.array([[dx1,dx2,dx3,dx4],[dy1,dy2,dy3,dy4]])
        bd_imf=draw.polygon2mask([ms,ns],np.array([[dy1,dy2,dy3,dy4],[dx1,dx2,dx3,dx4]]).T)
    return decp,bd_imf

def decpose_rec(pt_list_int,pt_list_float,line_list_int,polygon_line,img_dsm_t,img_ortho_t,L_mask,Td,Th1,Th2):
    L_num=len(pt_list_int)
    decp_rect=[]
    
    #range(L_num)
    for i in range(L_num):
    #for i in range(64,L_num):
        pt_int_temp=np.array(copy.deepcopy(pt_list_int[i]))
        pt_float_temp=np.array(copy.deepcopy(pt_list_float[i]))
        line_int_temp=np.array(copy.deepcopy(line_list_int[i]))
        if np.min(pt_int_temp) < 1 or np.max(pt_int_temp[:, 1]) > np.shape(img_dsm_t)[1] - 10 or np.max(
                pt_int_temp[:, 2]) > np.shape(img_dsm_t)[0] - 10:
            decp_rect.append([])
            continue
            
        #generate mask in local area
        pt_row=pt_int_temp[:,2]
        pt_col=pt_int_temp[:,1]
        range_local=np.array([min(pt_row)-10,max(pt_row)+10,min(pt_col)-10,max(pt_col)+10])
        pt_cent=np.array([np.mean(range_local[0:2])-range_local[0],np.mean(range_local[2:4])-range_local[2]])
        pt_local=np.array([pt_row-range_local[0],pt_col-range_local[2]]).T-pt_cent
        pt_int_loc=np.array([pt_row-range_local[0],pt_col-range_local[2]]).T
        mask_loc=[range_local[1]-range_local[0],range_local[3]-range_local[2]]
        mask_rec=draw.polygon2mask(mask_loc,pt_int_loc).astype(float)
        pt_local_ft=np.array([pt_float_temp[:,2]-range_local[0],pt_float_temp[:,1]-range_local[2]]).T-pt_cent
        imgsize=np.shape(L_mask)
        img_mask=np.zeros((imgsize[0],imgsize[1]),dtype=np.uint8)
        img_mask[np.where(L_mask==(i+1))]=1
        mask_height=img_mask*img_dsm_t
        
        #1. rotate lines and mask to main orienatation
        ori_line=np.array(copy.deepcopy(polygon_line[i]))
        sita_v=line_int_temp[np.argmax(np.array(ori_line)[:,7]),6]
        ori0=ori_line[np.argmax(np.array(ori_line)[:,7]),6]
        orie=[]
        ori_line1=copy.deepcopy(ori_line)
        
        for j, ori in enumerate(ori_line):
            if ori[6]==ori0:
                ori_line1[j,7]=1
            elif ori[6]==ori0-pi/2:
                ori_line1[j,7]=1
            elif ori[6]==ori0+pi/2:
                ori_line1[j,7]=1
        oriflag=0
        contlen=len(np.where(ori_line1[:,7]>1)[0])
        dsm_loc=copy.deepcopy(img_dsm_t[range_local[0]:range_local[1],range_local[2]:range_local[3]])
        ortho_loc=copy.deepcopy(img_ortho_t[range_local[0]:range_local[1],range_local[2]:range_local[3],:])
        min_height=mask_height.min()-5
        while contlen!=0:
            if max(ori_line[:,7])<10:
                break
            sita_v=line_int_temp[np.argmax(np.array(ori_line1)[:,7]),6]
            orie.append(ori_line[np.argmax(np.array(ori_line1)[:,7]),6])
            oriflag+=1
            for j, ori in enumerate(ori_line):
                if ori[6]==orie[oriflag-1]:
                    ori_line1[j,7]=1
                elif ori[6]==orie[oriflag-1]-pi/2:
                    ori_line1[j,7]=1
                elif ori[6]==orie[oriflag-1]+pi/2:
                    ori_line1[j,7]=1
            contlen=len(np.where(ori_line1[:,7]>1)[0])
        if len(orie)>0:
            orie=orie-ori0
            
        #calculate rotate matrix
        r1=np.array([[cos(ori0),-sin(ori0)],[sin(ori0),cos(ori0)]])
        r2=np.array([[cos(ori0),sin(ori0)],[-sin(ori0),cos(ori0)]])
        pt_loc0=np.array([pt_local[:,1],pt_local[:,0]]).T
        pt_loc_ft0=np.array([pt_local_ft[:,1],pt_local_ft[:,0]]).T
        pt_trans=np.dot(r2,pt_loc0.T).T
        pt_trans_m=pt_trans-np.array([min(pt_trans[:,0])-10,min(pt_trans[:,1])-10])
        pt_trans_ft=np.dot(r2,pt_loc_ft0.T).T
        pt_trans_ft_m=pt_trans_ft-np.array([min(pt_trans_ft[:,0])-10,min(pt_trans_ft[:,1])-10])
        height=ceil(max(pt_trans_m[:,1]))+10
        width=ceil(max(pt_trans_m[:,0]))+10
        
        #calculate and clip rotated DSM and Ortho
        bd_mask=draw.polygon2mask([height,width],np.array([pt_trans_ft_m[:,1],pt_trans_ft_m[:,0]]).T).astype(float)
        dsm_rot=rotate_image(dsm_loc,ori0)
        ortho_rot=rotate_image(ortho_loc,ori0)
        bd_rot=rotate_image(mask_rec,ori0).astype(int)
        (mloc_rot,nloc_rot)=np.shape(bd_rot)
        dsm_rot_ex=np.zeros((mloc_rot+20,nloc_rot+20),dtype=np.float32)
        ortho_rot_ex=np.zeros((mloc_rot+20,nloc_rot+20,3),dtype=np.uint)
        bd_rot_ex=np.zeros((mloc_rot+20,nloc_rot+20),dtype=np.uint)
        dsm_rot_ex[10:mloc_rot+10,10:nloc_rot+10]=dsm_rot
        ortho_rot_ex[10:mloc_rot+10,10:nloc_rot+10,:]=ortho_rot
        bd_rot_ex[10:mloc_rot+10,10:nloc_rot+10]=bd_rot
        (mloc_inf,nloc_inf)=np.where(bd_rot_ex==1)
        if np.sum(bd_mask)<10 or len(mloc_inf)<1:
            decp_rect.append([])
            continue
        nrot=min(nloc_inf)-10
        mrot=min(mloc_inf)-10
        dsm_mask = dsm_rot_ex[min(mrot, mloc_rot + 20 - height):mrot + height,min(nrot, nloc_rot + 20 - width):nrot + width]
        ortho_mask = ortho_rot_ex[min(mrot, mloc_rot + 20 - height):mrot + height,min(nrot, nloc_rot + 20 - width):nrot + width, :]
        
        #2. initial separation
        dsm_mask=dsm_mask*bd_mask
        ortho_mask[:,:,0]=ortho_mask[:,:,0]*bd_mask
        ortho_mask[:,:,1]=ortho_mask[:,:,1]*bd_mask
        ortho_mask[:,:,2]=ortho_mask[:,:,2]*bd_mask
        (ty,tx)=np.where(bd_mask==1)
        
        #direction for vertical (from up to down)
        ort_dv=[]
        dsm_dv=[]
        for j in range(max(ty) - min(ty)):
            ort_dv.append(sum(ortho_mask[min(ty) + j, :, :]) / sum(bd_mask[min(ty) + j, :]))
            dsm_dv.append(sum(dsm_mask[min(ty) + j, :]) / sum(bd_mask[min(ty) + j, :]))
        if len(dsm_dv) > 2:
            ort_dvf = np.mean(abs(np.gradient(np.array(ort_dv))[0]), axis=1)
            dsm_dvf = abs(np.gradient(np.array(dsm_dv).T).T)
        else:
            ort_dvf = []
            dsm_dvf = []
        if len(ort_dvf) < 1:
            sp_idx = []
            sp_idxv = []
        else:
            sp_idxd=list(signal.find_peaks(-np.array(dsm_dv),distance=10)[0])
            sp_idxd.extend(list(np.where(dsm_dvf>0.2)[0]))
            sp_idxor=list(signal.find_peaks(ort_dvf,height=10,distance=10)[0])
            sp_idxor1=list(np.where(ort_dvf[sp_idxor]>50)[0]) #50
            sp_idx=list((set(sp_idxor).intersection(set(sp_idxd))).union(set(sp_idxor1).intersection(set(sp_idxor))))
            sp_idxv=[]
            for j in range(len(np.where(np.array(sp_idx)<10)[0]),0,-1):
                sp_idx.pop(j-1)
            for j in range(len(np.where(np.array(sp_idx)>max(tx)-min(tx)-10)[0]),0,-1):
                sp_idx.pop(j-1)
            if len(sp_idx)>0:
                for j, idx in enumerate(sp_idx):
                    diff_mask=copy.deepcopy(bd_mask)
                    diff_mask[idx+min(ty),:]=0
                    diff_mask=bd_mask-diff_mask
                    (typ,txp)=np.where(diff_mask==1)
                    tbd_up=0*bd_mask
                    tbd_down=0*bd_mask
                    tbd_up[typ[0]-3:typ[0],min(txp):max(txp)]=1
                    tbd_down[typ[0]:typ[0]+3,min(txp):max(txp)]=1
                    mean_ort_up=np.array([np.sum(tbd_up*ortho_mask[:,:,0]),np.sum(tbd_up*ortho_mask[:,:,1]),\
                                     np.sum(tbd_up*ortho_mask[:,:,2])])/len(np.where(tbd_up*ortho_mask[:,:,0]>1)[0])
                    grad_dsm_up=np.sum(dsm_mask[typ[0]-3,min(txp):max(txp)])/len(np.where(dsm_mask[typ[0]-3,min(txp):max(txp)]>1)[0])\
                        -np.sum(dsm_mask[typ[0],min(txp):max(txp)])/len(np.where(dsm_mask[typ[0],min(txp):max(txp)]>1)[0])
                    mean_ort_down=np.array([np.sum(tbd_down*ortho_mask[:,:,0]),np.sum(tbd_down*ortho_mask[:,:,1]),\
                                     np.sum(tbd_down*ortho_mask[:,:,2])])/len(np.where(tbd_down*ortho_mask[:,:,0]>1)[0])
                    grad_dsm_down=np.sum(dsm_mask[typ[0],min(txp):max(txp)])/len(np.where(dsm_mask[typ[0],min(txp):max(txp)]>1)[0])\
                        -np.sum(dsm_mask[typ[0]+3,min(txp):max(txp)])/len(np.where(dsm_mask[typ[0]+3,min(txp):max(txp)]>1)[0])
                    if (abs(np.mean(mean_ort_up-mean_ort_down))>5 or abs(grad_dsm_up-grad_dsm_down)>0.5)and max(np.diff(txp))==1:
                        sp_idxv.append(idx+min(ty))
            
        #direction for horizontal (from left to right)
        ort_dh=[]
        dsm_dh=[]
        for j in range(max(tx)-min(tx)):
            ort_dh.append(sum(ortho_mask[:,min(tx)+j,:])/sum(bd_mask[:,min(tx)+j]))
            dsm_dh.append(sum(dsm_mask[:,min(tx)+j])/sum(bd_mask[:,min(tx)+j]))
        if len(dsm_dh) > 2:
            ort_dhf = np.mean(abs(np.gradient(np.array(ort_dh))[0]), axis=1)
            dsm_dhf = abs(np.gradient(np.array(dsm_dh).T).T)
        else:
            ort_dhf = []
            dsm_dhf = []
        if len(ort_dhf) < 1:
            sp_idx = []
            sp_idxh = []
        else:
            sp_idxd=list(signal.find_peaks(-np.array(dsm_dh),distance=10)[0])
            sp_idxd.extend(list(np.where(dsm_dhf>0.2)[0]))
            sp_idxor=list(signal.find_peaks(ort_dhf,height=10,distance=10)[0])
            sp_idxor1=list(np.where(np.array(ort_dhf)>50)[0])
            sp_idx=list((set(sp_idxor).intersection(set(sp_idxd))).union(set(sp_idxor1).intersection(set(sp_idxor))))
            sp_idxh=[]

            for j in range(len(np.where(np.array(sp_idx)<10)[0]),0,-1):
                sp_idx.pop(j-1)
            for j in range(len(np.where(np.array(sp_idx)>max(ty)-min(ty)-10)[0]),0,-1):
                sp_idx.pop(j-1)
            if len(sp_idx)>0:
                for j, idx in enumerate(sp_idx):
                    diff_mask=copy.deepcopy(bd_mask)
                    diff_mask[:,idx+min(tx)]=0
                    diff_mask=bd_mask-diff_mask
                    (typ,txp)=np.where(diff_mask==1)
                    tbd_left=0*bd_mask
                    tbd_right=0*bd_mask
                    if len(typ) < 1:
                        continue
                    tbd_left[min(typ):max(typ),txp[0]-3:txp[0]]=1
                    tbd_right[min(typ):max(typ),txp[0]-3:txp[0]]=1
                    mean_ort_left=np.array([np.sum(tbd_left*ortho_mask[:,:,0]),np.sum(tbd_left*ortho_mask[:,:,1]),\
                                     np.sum(tbd_left*ortho_mask[:,:,2])])/len(np.where(tbd_left*ortho_mask[:,:,0]>1)[0])
                    grad_dsm_left=np.sum(dsm_mask[min(typ):max(typ),txp[0]-3])/len(np.where(dsm_mask[min(typ):max(typ),txp[0]-3]>1)[0])\
                        -np.sum(dsm_mask[min(typ):max(typ),txp[0]])/len(np.where(dsm_mask[min(typ):max(typ),txp[0]]>1)[0])
                    mean_ort_right=np.array([np.sum(tbd_right*ortho_mask[:,:,0]),np.sum(tbd_right*ortho_mask[:,:,1]),\
                                     np.sum(tbd_right*ortho_mask[:,:,2])])/len(np.where(tbd_right*ortho_mask[:,:,0]>1)[0])
                    grad_dsm_right=np.sum(dsm_mask[min(typ):max(typ),txp[0]])/len(np.where(dsm_mask[min(typ):max(typ),txp[0]]>1)[0])\
                        -np.sum(dsm_mask[min(typ):max(typ),txp[0]+3])/len(np.where(dsm_mask[min(typ):max(typ),txp[0]+3]>1)[0])
                    if (abs(np.mean(mean_ort_left-mean_ort_right))>5 or abs(grad_dsm_left-grad_dsm_right)>0.5)and max(np.diff(typ))==1:
                        sp_idxh.append(idx+min(tx))
        
        #3. generate pyramid maximum inner rectangle in each grid level
        #generate pyramid mask
        bd_mask_t=copy.deepcopy(bd_mask)
        im_4=cv2.resize(bd_mask_t,None,fx=0.25,fy=0.25).astype(int)
        im_2=cv2.resize(bd_mask_t,None,fx=0.5,fy=0.5).astype(int)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        im_40 = cv2.morphologyEx(im_4.astype(float),cv2.MORPH_CLOSE,kernel).astype(int)
        if len(sp_idxv)>0:
            bd_mask_t[sp_idxv,:]=0
            im_4[[round(k/4) for k in sp_idxv],:]=0
            im_40[[round(k/4) for k in sp_idxv],:]=0
            im_2[[round(k/2) for k in sp_idxv],:]=0
        if len(sp_idxh)>0:
            bd_mask_t[:,sp_idxh]=0
            im_4[:,[round(k/4) for k in sp_idxh]]=0
            im_40[:,[round(k/4) for k in sp_idxh]]=0
            im_2[:,[round(k/2) for k in sp_idxh]]=0
        im_1=bd_mask_t.astype(int)
        
        #get maximum rectangles
        count_im4=np.sum(im_4)
        (m4,n4)=np.shape(im_4)
        (m2,n2)=np.shape(im_2)
        (m1,n1)=np.shape(im_1)
        decp4=[]
        decp2=[]
        decp1=[]
        
        recori=[]
        flagnum=0
        min_area=4
        
        #grid level 1
        while count_im4>min_area and flagnum<20:
            LRout = findRotMaxRect((1-im_40), flag_opt=True, main_angle=0)
            for j, lrout in enumerate(LRout):
                lrout[0],lrout[1]=lrout[1],lrout[0]
            areaLR = (sqrt((LRout[0][0] - LRout[1][0]) ** 2 + (LRout[0][1] - LRout[1][1]) ** 2) + 1) * \
                     (sqrt((LRout[0][0] - LRout[3][0]) ** 2 + (LRout[0][1] - LRout[3][1]) ** 2) + 1)
            LRout_sd=[[0,0],[0,0],[0,0],[0,0]]
            rec=np.array(LRout).T
            LRout_list=[]
            arealr=[]
            for j in range(len(orie)):
                if abs(sum(rec[:,0])-sum(rec[:,1]))>1 and abs(sum(rec[:,0])-sum(rec[:,3]))>1:
                    LRout_sd=findRotMaxRect((1-im_40), flag_opt=True, main_angle=orie[j])
                    for k, lrout in enumerate(LRout_sd):
                        lrout[0],lrout[1]=lrout[1],lrout[0]
                else:
                    LRout_sd=[[0,0],[0,0],[0,0],[0,0]]
                arealr.append((sqrt((LRout_sd[0][0] - LRout_sd[1][0] + 1) ** 2 + (LRout_sd[0][1] - LRout_sd[1][1] + 1) ** 2) + 1) * \
                              (sqrt((LRout_sd[0][0] - LRout_sd[3][0] + 1) ** 2 + (LRout_sd[0][1] - LRout_sd[3][1] + 1) ** 2) + 1))
                LRout_list.append(LRout_sd)
            if len(arealr)>0:
                arealr_max=max(arealr)
                ori_num=arealr.index(arealr_max)
            if len(orie)<1:
                arealr_max=0
                ori_num=-1
            flagnum+=1
            if areaLR>=arealr_max and areaLR>min_area:
                im_40[int(LRout[0][1]):int(LRout[2][1]),int(LRout[0][0]):int(LRout[2][0])]=0
                count_im4-=areaLR
                if abs(sum(rec[:,0])-sum(rec[:,1]))<1 or abs(sum(rec[:,0])-sum(rec[:,3]))<1 or max(areaLR,arealr_max)<min_area:
                    rec=[]
                else:
                    recori.append(0)
            else:
                if ori_num==-1 or areaLR<min_area:
                    continue
                LRout_sd=LRout_list[ori_num]
                txy=np.array(LRout_sd)
                im_40t=draw.polygon2mask([m4,n4],np.array([txy[:,1],txy[:,0]]).T)
                im_40=im_40-im_40t
                rec=txy.T
                count_im4 -= (sqrt((LRout_sd[0][0] - LRout_sd[1][0]) ** 2 + (LRout_sd[0][1] - LRout_sd[1][1]) ** 2) + 1) * \
                             (sqrt((LRout_sd[0][0] - LRout_sd[3][0]) ** 2 + (LRout_sd[0][1] - LRout_sd[3][1]) ** 2) + 1)
                recori.append(orie[ori_num])
            decp4.append(rec)
        decp4 = [x for x in decp4 if x != []]
        decp_num0=len(decp4)
        if decp_num0==0:
            decp_rect.append([])
            continue
            
        #grid level 2
        im_20=copy.deepcopy(im_2)
        for j, decp in enumerate(decp4):
            rec=np.array(decp).T
            rec0=copy.deepcopy(rec)
            rec0[np.where(rec0[:,0]==min(rec0[:,0])),0]=min(rec0[:,0])-1
            rec0[np.where(rec0[:,1]==min(rec0[:,1])),1]=min(rec0[:,1])-1
            bd_im4=draw.polygon2mask([m4,n4],np.array([rec0[:,1],rec0[:,0]]).T).astype(float)
            (recr0,recc0)=np.where(bd_im4==1)
            if len(recr0) < 1:
                continue
            if min(recr0)==max(recr0):
                if rec[0,1]!=rec[2,1]:
                    bd_im4=draw.polygon2mask([m4,n4],np.array([np.array([rec[0,1]-1,rec[1,1]-1,rec[2,1],rec[3,1]]).T,rec[:,0]]).T).astype(float)
                else:
                    bd_im4=draw.polygon2mask([m4,n4],np.array([np.array([rec[0,1]-1,rec[1,1],rec[2,1]-1,rec[3,1]]).T,rec[:,0]]).T).astype(float)
            bd_im4r_s2=cv2.resize(bd_im4,(n2,m2)).astype(int)
            
            decp,bd_im2f=getgridmask(bd_im4r_s2,recori[j],im_20)
            
            decp2.append(decp)
            im_20=im_20-bd_im2f
        
        #grid level 3
        decp2 = [x for x in decp2 if x != []]
        im_10=copy.deepcopy(im_1)
        for j, decp in enumerate(decp2):
            rec=np.array(decp).T
            rec0=copy.deepcopy(rec)
            rec0[np.where(rec0[:,0]==min(rec0[:,0])),0]=min(rec0[:,0])-1
            rec0[np.where(rec0[:,1]==min(rec0[:,1])),1]=min(rec0[:,1])-1
            bd_im2=draw.polygon2mask([m2,n2],np.array([rec0[:,1],rec0[:,0]]).T).astype(float)
            (recr0,recc0)=np.where(bd_im2==1)
            if len(recr0)<1:
                continue
            if min(recr0)==max(recr0):
                if rec[0,1]!=rec[2,1]:
                    bd_im2=draw.polygon2mask([m2,n2],np.array([np.array([rec[0,1]-1,rec[1,1]-1,rec[2,1],rec[3,1]]).T,rec[:,0]]).T).astype(float)
                else:
                    bd_im2=draw.polygon2mask([m2,n2],np.array([np.array([rec[0,1]-1,rec[1,1],rec[2,1]-1,rec[3,1]]).T,rec[:,0]]).T).astype(float)
            bd_im2r_s1=cv2.resize(bd_im2,(n1,m1)).astype(int)
            
            decp,bd_im1f=getgridmask(bd_im2r_s1,recori[j],im_10)

            if len(decp) > 0:
                decp1.append(decp)
            im_10=im_10-bd_im1f   
              
        #4. seperation and merging
        decp1 = [x for x in decp1 if x != []]
        decp_num0 = len(decp1)
        if decp_num0>1:
            #seperate
            spdecp=[]
            for j in range(decp_num0):
                rec=np.array(decp1[j]).T.astype(int)
                bd_im1=draw.polygon2mask([m1,n1],np.array([np.array([rec[0,1]-1,rec[1,1]-1,rec[2,1],rec[3,1]]).T,rec[:,0]]).T).astype(int)
                area_bd=np.sum(bd_im1)
                if area_bd>1000 and recori[j]==0:
                    centd=np.array([ceil((rec[0,0]+rec[1,0])/2),ceil((rec[1,1]+rec[2,1])/2)])
                    newrec=[]
                    if abs(rec[0,0]-rec[1,0])>abs(rec[1,1]-rec[2,1]):
                        grad_dsm=np.gradient(dsm_mask[ceil((rec[1,1]+rec[2,1])/2),rec[0,0]:rec[1,0]])
                        sp_idx=list(signal.find_peaks(abs(grad_dsm),height=0.25,distance=10)[0])
                        for q in range(len(np.where(np.array(sp_idx)<10)[0]),0,-1):
                            sp_idx.pop(q-1)
                        for q in range(len(np.where(np.array(sp_idx)>len(grad_dsm)-10)[0]),0,-1):
                            sp_idx.pop(q-1)
                        sp_idx0=[]
                        if len(sp_idx)>0:
                            for k in range(len(sp_idx)):
                                ts_mask=ortho_mask[centd[1]-10:centd[1]+10,rec[0,0]+sp_idx[k]-10:rec[0,0]+sp_idx[k]+10,:]
                                tsr, tsg, tsb=ts_mask[:,:,0], ts_mask[:,:,1], ts_mask[:,:,2]
                                avg_std_rgb=np.mean(np.array([np.std(tsr[np.where(tsr>1)]),\
                                                              np.std(tsg[np.where(tsg>1)]),np.std(tsb[np.where(tsb>1)])]))
                                flagsp=1
                                if len(sp_idx0)>0:
                                    flagsp=sp_idx0[len(sp_idx0)-1]
                                if avg_std_rgb>12 and abs(flagsp-sp_idx[k])>10:
                                    sp_idx0.append(sp_idx[k])
                            prex=rec[0,0]
                            sp_idx0.append(len(grad_dsm))
                            if len(sp_idx0)>1:
                                for k in range(len(sp_idx0)):
                                    newrec.append(np.array([[prex,rec[0,0]+sp_idx0[k],rec[0,0]+sp_idx0[k],prex],\
                                                        [rec[0,1],rec[1,1],rec[2,1],rec[3,1]]]))
                                    prex=rec[0,0]+sp_idx0[k]
                            else:
                                newrec.append(decp1[j])
                            #newrec=np.array(newrec)
                        else:
                            newrec.append(decp1[j])
                    else:
                        grad_dsm=np.gradient(dsm_mask[rec[1,1]:rec[2,1],ceil((rec[0,0]+rec[1,0])/2)])
                        sp_idx=list(signal.find_peaks(abs(grad_dsm),height=0.25,distance=10)[0])
                        for q in range(len(np.where(np.array(sp_idx)<10)[0]),0,-1):
                            sp_idx.pop(q-1)
                        for q in range(len(np.where(np.array(sp_idx)>len(grad_dsm)-10)[0]),0,-1):
                            sp_idx.pop(q-1)
                        sp_idx0=[]
                        if len(sp_idx)>0:
                            for k in range(len(sp_idx)):
                                ts_mask=ortho_mask[rec[0,1]+sp_idx[k]-10:rec[0,1]+sp_idx[k]+10,centd[1]-10:centd[1]+10,:]
                                tsr, tsg, tsb=ts_mask[:,:,0], ts_mask[:,:,1], ts_mask[:,:,2]
                                avg_std_rgb=np.mean(np.array([np.std(tsr[np.where(tsr>1)]),\
                                                              np.std(tsg[np.where(tsg>1)]),np.std(tsb[np.where(tsb>1)])]))
                                flagsp=1
                                if len(sp_idx0)>0:
                                    flagsp=sp_idx0[len(sp_idx0)-1]
                                if avg_std_rgb>12 and abs(flagsp-sp_idx[k])>10:
                                    sp_idx0.append(sp_idx[k])
                            prey=rec[0,1]
                            sp_idx0.append(len(grad_dsm))
                            if len(sp_idx0)>1:
                                for k in range(len(sp_idx0)):
                                    newrec.append(np.array([[rec[0,0],rec[1,0],rec[2,0],rec[3,0]],\
                                                            [prey,prey,rec[0,1]+sp_idx0[k],rec[0,1]+sp_idx0[k]]]))
                                    prey=rec[0,1]+sp_idx0[k]
                            else:
                                newrec.append(decp1[j])
                            #newrec=np.array(newrec)
                        else:
                            newrec.append(decp1[j])
                    spdecp.extend(newrec)
                else:
                    spdecp.append(decp1[j])
                
            #merge
            sp_num=len(spdecp)
            kernel_m=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            mean_rgb=np.zeros((sp_num,3),dtype=float)
            std_rgb=np.zeros((sp_num,3),dtype=float)
            mean_dsm=np.zeros((sp_num,1),dtype=float)
            cent_bd=np.zeros((sp_num,2),dtype=float)
            area_bd=np.zeros((sp_num,1),dtype=float)
            mgdecp=[]
            for j in range(sp_num):
                rec=np.array(spdecp[j]).T
                bd_im1=draw.polygon2mask([m1,n1],np.array([np.array([rec[0,1]-1,rec[1,1]-1,rec[2,1],rec[3,1]]).T,rec[:,0]]).T).astype(int)
                area_bd[j]=np.sum(bd_im1)
                mask_r=bd_im1*ortho_mask[:,:,1]
                mask_g=bd_im1*ortho_mask[:,:,1]
                mask_b=bd_im1*ortho_mask[:,:,1]
                mean_rgb[j,:]=np.array([mask_r.sum(),mask_g.sum(),mask_b.sum()])/len(np.where(mask_r>1)[0])
                std_rgb[j,:]=np.array([np.std(mask_r[np.where(mask_r>1)]),np.std(mask_g[np.where(mask_g>1)]),\
                                       np.std(mask_b[np.where(mask_b>1)])])
                mean_dsm[j]=np.sum(bd_im1*dsm_mask)/area_bd[j]
                cent_bd[j,:]=np.array([ceil((rec[0,0]+rec[1,0])/2),ceil((rec[1,1]+rec[2,1])/2)])
            flagmat=np.triu(np.ones((sp_num,sp_num),dtype=int))+np.diag(np.ones((sp_num,sp_num),dtype=int))
            for j in range(sp_num):
                rec=np.array(spdecp[j]).T.astype(int)
                rec0=copy.deepcopy(rec)
                rec0[np.where(rec0[:,0]==min(rec0[:,0])),0]=min(rec0[:,0])-1
                rec0[np.where(rec0[:,1]==min(rec0[:,1])),1]=min(rec0[:,1])-1
                bd_im1=draw.polygon2mask([m1,n1],np.array([rec0[:,1],rec0[:,0]]).T).astype(int)
                inlrec=cv2.dilate(bd_im1.astype(float),kernel_m,iterations=1).astype(int)
                for k in range(sp_num):
                    if flagmat[j,k]!=1:
                        continue
                    rec1=np.array(spdecp[k]).T
                    rec0=copy.deepcopy(rec1)
                    rec0[np.where(rec0[:,0]==min(rec0[:,0])),0]=min(rec0[:,0])-1
                    rec0[np.where(rec0[:,1]==min(rec0[:,1])),1]=min(rec0[:,1])-1
                    bd_im1cp=draw.polygon2mask([m1,n1],np.array([rec0[:,1],rec0[:,0]]).T).astype(int)
                    inlreccp=cv2.dilate(bd_im1cp.astype(float),kernel_m,iterations=1).astype(int)
                    dm_rgb=np.mean(abs(mean_rgb[j,:]-mean_rgb[k,:]))
                    ds_rgb=np.mean(abs(std_rgb[j,:]-std_rgb[k,:]))
                    dm_dsm=abs(mean_dsm[j]-mean_dsm[k])
                    dxy=cent_bd[j,:]-cent_bd[k,:]
                    bd_int=cv2.dilate(inlrec*inlreccp.astype(float),kernel_m,iterations=1).astype(int)
                    mask_r, mask_g, mask_b=bd_int*ortho_mask[:,:,0],bd_int*ortho_mask[:,:,1],bd_int*ortho_mask[:,:,2]
                    mean_rgb_int=np.array([mask_r.sum(),mask_g.sum(),mask_b.sum()])/len(np.where(mask_r>1)[0])
                    std_rgb_int=np.array([np.std(mask_r[np.where(mask_r>1)]),np.std(mask_g[np.where(mask_g>1)]),\
                                       np.std(mask_b[np.where(mask_b>1)])])
                    mean_dsm_int=np.sum(bd_int*dsm_mask)/np.sum(bd_int)
                    dm_int=abs(mean_dsm_int-(mean_dsm[j]-mean_dsm[k]))
                    itmrgb=abs(np.mean(mean_rgb_int-abs(mean_rgb[j,:]+mean_rgb[k,:])/2))
                    itsrgb=abs(np.mean(std_rgb_int-abs(std_rgb[j,:]+std_rgb[k,:])/2))
                    if itmrgb<Td and itsrgb<20 and dm_int<Th1:
                        tagmerge=(dm_rgb<5 and ds_rgb<15 and dm_dsm<Th1 and min(abs(dxy))<5)\
                            or (((min(itmrgb,itsrgb)<Td and max(itmrgb,itsrgb)<8) or (itmrgb<Td/2 and itsrgb<10))\
                                and dm_int<Th2*2 and min(abs(dxy))<5)
                        if tagmerge==1:
                            if abs(dxy[0])>abs(dxy[1]):
                                dy=floor((cent_bd[j,1]*area_bd[j]+cent_bd[k,1]*area_bd[k])/(area_bd[j]+area_bd[k]))
                                recxy1=min(min(rec[:,0]),min(rec1[:,0]))
                                recxy2=max(max(rec[:,0]),max(rec1[:,0]))
                                recxy3=np.mean(np.array([min(rec[:,1])-(cent_bd[j,1]-dy),min(rec1[:,1])-(cent_bd[k,1]-dy)]))
                                recxy4=np.mean(np.array([max(rec[:,1])-(cent_bd[j,1]-dy),max(rec1[:,1])-(cent_bd[k,1]-dy)]))
                                mgdecp.append(np.array([[recxy1,recxy2,recxy2,recxy1],[recxy3,recxy3,recxy4,recxy4]]))
                            else:
                                dx=floor((cent_bd[j,0]*area_bd[j]+cent_bd[k,0]*area_bd[k])/(area_bd[j]+area_bd[k]))
                                recxy1=np.mean(np.array([min(rec[:,0])-(cent_bd[j,0]-dx),min(rec1[:,0])-(cent_bd[k,0]-dx)]))
                                recxy2=np.mean(np.array([max(rec[:,0])-(cent_bd[j,0]-dx),max(rec1[:,0])-(cent_bd[k,0]-dx)]))
                                recxy3=min(min(rec[:,1]),min(rec1[:,1]))
                                recxy4=max(max(rec[:,1]),max(rec1[:,1]))
                                mgdecp.append(np.array([[recxy1,recxy2,recxy2,recxy1],[recxy3,recxy3,recxy4,recxy4]]))
                            flagmat[j,:]=0
                            flagmat[:,k]=0
                            flagmat[k,:]=0
                if max(flagmat[j,:])==2:
                    mgdecp.append(spdecp[j])
        else:
            mgdecp=copy.deepcopy(decp1)
        
        #5. transfer coordinate to whole image
        jflag=1
        mgdecp_num=len(mgdecp)
        decp_rec=[]
        for j in range(mgdecp_num):
            recp=np.array(mgdecp[j]).T+np.array([min(pt_trans[:,0])-10,min(pt_trans[:,1])-10])
            pr_trans0=np.dot(r1,recp.T).T
            pr_trans0[:,0]+=pt_cent[1]
            pr_trans0[:,1]+=pt_cent[0]
            pr_trans=copy.deepcopy(pr_trans0)
            pr_trans[:,0]=pr_trans0[:,0]+range_local[2]
            pr_trans[:,1]=pr_trans0[:,1]+range_local[0]
            tx=np.array([round(k) for k in pr_trans0[:,0]])
            ty=np.array([round(k) for k in pr_trans0[:,1]])
            tx[np.where(tx<1)]=1
            ty[np.where(ty<1)]=1
            mask1=np.zeros((mask_loc[0],mask_loc[1]),dtype=int)
            mask0=draw.polygon2mask([mask_loc[0],mask_loc[1]],np.array([ty,tx]).T).astype(int)
            area0=np.sum(mask0)
            mask1[min(ty):max(ty),min(tx):max(tx)]=img_mask[min(ty)+range_local[0]:max(ty)+range_local[0],\
                                                            min(tx)+range_local[2]:max(tx)+range_local[2]]
            over2d=np.sum(mask1*mask0)/area0
            areadecp=abs((mgdecp[j][0,2]-mgdecp[j][0,0])*(mgdecp[j][1,2]-mgdecp[j][1,0]))
            #if over2d>0.7:
            decp_rec.append([i,jflag,pr_trans[0,0],pr_trans[0,1],pr_trans[1,0],pr_trans[1,1]\
                            ,pr_trans[3,0],pr_trans[3,1],pr_trans[2,0],pr_trans[2,1],int(areadecp),sita_v])
            jflag+=1
        decp_rect.append(decp_rec)

    return decp_rect

def decpose_rec_multi(polygon_line, pt_list_int, pt_list_float, line_list_int, img_dsm_t, img_ortho_t, L_mask, Td, Th1, Th2):
    L_num = len(pt_list_int)
    decp_rect = []
    i = polygon_line[0][0]



    pt_int_temp = np.array(copy.deepcopy(pt_list_int[i]))
    pt_float_temp = np.array(copy.deepcopy(pt_list_float[i]))
    line_int_temp = np.array(copy.deepcopy(line_list_int[i]))
    if np.min(pt_int_temp) < 1 or np.max(pt_int_temp[:, 1]) > np.shape(img_dsm_t)[1] - 10 or np.max(
            pt_int_temp[:, 2]) > np.shape(img_dsm_t)[0] - 10:
        return(decp_rect)

    # generate mask in local area
    pt_row = pt_int_temp[:, 2]
    pt_col = pt_int_temp[:, 1]
    range_local = np.array([min(pt_row) - 10, max(pt_row) + 10, min(pt_col) - 10, max(pt_col) + 10])
    pt_cent = np.array([np.mean(range_local[0:2]) - range_local[0], np.mean(range_local[2:4]) - range_local[2]])
    pt_local = np.array([pt_row - range_local[0], pt_col - range_local[2]]).T - pt_cent
    pt_int_loc = np.array([pt_row - range_local[0], pt_col - range_local[2]]).T
    mask_loc = [range_local[1] - range_local[0], range_local[3] - range_local[2]]
    mask_rec = draw.polygon2mask(mask_loc, pt_int_loc).astype(float)
    pt_local_ft = np.array([pt_float_temp[:, 2] - range_local[0], pt_float_temp[:, 1] - range_local[2]]).T - pt_cent
    imgsize = np.shape(L_mask)
    img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
    img_mask[np.where(L_mask == (i + 1))] = 1
    mask_height = img_mask * img_dsm_t

    # 1. rotate lines and mask to main orienatation
    ori_line = np.array(copy.deepcopy(polygon_line))
    sita_v = line_int_temp[np.argmax(np.array(ori_line)[:, 7]), 6]
    ori0 = ori_line[np.argmax(np.array(ori_line)[:, 7]), 6]
    orie = []
    ori_line1 = copy.deepcopy(ori_line)

    for j, ori in enumerate(ori_line):
        if ori[6] == ori0:
            ori_line1[j, 7] = 1
        elif ori[6] == ori0 - pi / 2:
            ori_line1[j, 7] = 1
        elif ori[6] == ori0 + pi / 2:
            ori_line1[j, 7] = 1
    oriflag = 0
    contlen = len(np.where(ori_line1[:, 7] > 1)[0])
    dsm_loc = copy.deepcopy(img_dsm_t[range_local[0]:range_local[1], range_local[2]:range_local[3]])
    ortho_loc = copy.deepcopy(img_ortho_t[range_local[0]:range_local[1], range_local[2]:range_local[3], :])
    min_height = mask_height.min() - 5
    while contlen != 0:
        if max(ori_line[:, 7]) < 10:
            break
        sita_v = line_int_temp[np.argmax(np.array(ori_line1)[:, 7]), 6]
        orie.append(ori_line[np.argmax(np.array(ori_line1)[:, 7]), 6])
        oriflag += 1
        for j, ori in enumerate(ori_line):
            if ori[6] == orie[oriflag - 1]:
                ori_line1[j, 7] = 1
            elif ori[6] == orie[oriflag - 1] - pi / 2:
                ori_line1[j, 7] = 1
            elif ori[6] == orie[oriflag - 1] + pi / 2:
                ori_line1[j, 7] = 1
        contlen = len(np.where(ori_line1[:, 7] > 1)[0])
    if len(orie) > 0:
        orie = orie - ori0

    # calculate rotate matrix
    r1 = np.array([[cos(ori0), -sin(ori0)], [sin(ori0), cos(ori0)]])
    r2 = np.array([[cos(ori0), sin(ori0)], [-sin(ori0), cos(ori0)]])
    pt_loc0 = np.array([pt_local[:, 1], pt_local[:, 0]]).T
    pt_loc_ft0 = np.array([pt_local_ft[:, 1], pt_local_ft[:, 0]]).T
    pt_trans = np.dot(r2, pt_loc0.T).T
    pt_trans_m = pt_trans - np.array([min(pt_trans[:, 0]) - 10, min(pt_trans[:, 1]) - 10]);
    pt_trans_ft = np.dot(r2, pt_loc_ft0.T).T
    pt_trans_ft_m = pt_trans_ft - np.array([min(pt_trans_ft[:, 0]) - 10, min(pt_trans_ft[:, 1]) - 10]);
    height = ceil(max(pt_trans_m[:, 1])) + 10;
    width = ceil(max(pt_trans_m[:, 0])) + 10;

    # calculate and clip rotated DSM and Ortho
    bd_mask = draw.polygon2mask([height, width], np.array([pt_trans_ft_m[:, 1], pt_trans_ft_m[:, 0]]).T).astype(
        float)
    dsm_rot = rotate_image(dsm_loc, ori0)
    ortho_rot = rotate_image(ortho_loc, ori0)
    bd_rot = rotate_image(mask_rec, ori0).astype(int)
    (mloc_rot, nloc_rot) = np.shape(bd_rot)
    dsm_rot_ex = np.zeros((mloc_rot + 20, nloc_rot + 20), dtype=np.float32)
    ortho_rot_ex = np.zeros((mloc_rot + 20, nloc_rot + 20, 3), dtype=np.uint)
    bd_rot_ex = np.zeros((mloc_rot + 20, nloc_rot + 20), dtype=np.uint)
    dsm_rot_ex[10:mloc_rot + 10, 10:nloc_rot + 10] = dsm_rot
    ortho_rot_ex[10:mloc_rot + 10, 10:nloc_rot + 10, :] = ortho_rot
    bd_rot_ex[10:mloc_rot + 10, 10:nloc_rot + 10] = bd_rot
    (mloc_inf, nloc_inf) = np.where(bd_rot_ex == 1)
    if np.sum(bd_mask) < 10 or len(mloc_inf) < 1:
        return(decp_rect)
    nrot = min(nloc_inf) - 10
    mrot = min(mloc_inf) - 10
    dsm_mask = dsm_rot_ex[min(mrot, mloc_rot + 20 - height):mrot + height,
               min(nrot, nloc_rot + 20 - width):nrot + width]
    ortho_mask = ortho_rot_ex[min(mrot, mloc_rot + 20 - height):mrot + height,
                 min(nrot, nloc_rot + 20 - width):nrot + width, :]

    # 2. initial separation
    dsm_mask = dsm_mask * bd_mask
    ortho_mask[:, :, 0] = ortho_mask[:, :, 0] * bd_mask
    ortho_mask[:, :, 1] = ortho_mask[:, :, 1] * bd_mask
    ortho_mask[:, :, 2] = ortho_mask[:, :, 2] * bd_mask
    (ty, tx) = np.where(bd_mask == 1)

    # direction for vertical (from up to down)
    ort_dv = []
    dsm_dv = []
    for j in range(max(ty) - min(ty)):
        ort_dv.append(sum(ortho_mask[min(ty) + j, :, :]) / sum(bd_mask[min(ty) + j, :]))
        dsm_dv.append(sum(dsm_mask[min(ty) + j, :]) / sum(bd_mask[min(ty) + j, :]))
    if len(dsm_dv) > 2:
        ort_dvf = np.mean(abs(np.gradient(np.array(ort_dv))[0]), axis=1)
        dsm_dvf = abs(np.gradient(np.array(dsm_dv).T).T)
    else:
        ort_dvf = []
        dsm_dvf = []
    if len(ort_dvf) < 1:
        sp_idx = []
        sp_idxv = []
    else:
        sp_idxd = list(signal.find_peaks(-np.array(dsm_dv), distance=10)[0])
        sp_idxd.extend(list(np.where(dsm_dvf > 0.2)[0]))
        sp_idxor = list(signal.find_peaks(ort_dvf, height=10, distance=10)[0])
        sp_idxor1 = list(np.where(ort_dvf[sp_idxor] > 50)[0])
        sp_idx = list((set(sp_idxor).intersection(set(sp_idxd))).union(set(sp_idxor1).intersection(set(sp_idxor))))
        sp_idxv = []
        for j in range(len(np.where(np.array(sp_idx) < 10)[0]), 0, -1):
            sp_idx.pop(j - 1)
        for j in range(len(np.where(np.array(sp_idx) > max(tx) - min(tx) - 10)[0]), 0, -1):
            sp_idx.pop(j - 1)
        if len(sp_idx) > 0:
            for j, idx in enumerate(sp_idx):
                diff_mask = copy.deepcopy(bd_mask)
                diff_mask[idx + min(ty), :] = 0
                diff_mask = bd_mask - diff_mask
                (typ, txp) = np.where(diff_mask == 1)
                tbd_up = 0 * bd_mask
                tbd_down = 0 * bd_mask
                tbd_up[typ[0] - 3:typ[0], min(txp):max(txp)] = 1
                tbd_down[typ[0]:typ[0] + 3, min(txp):max(txp)] = 1
                mean_ort_up = np.array([np.sum(tbd_up * ortho_mask[:, :, 0]), np.sum(tbd_up * ortho_mask[:, :, 1]), \
                                        np.sum(tbd_up * ortho_mask[:, :, 2])]) / len(
                    np.where(tbd_up * ortho_mask[:, :, 0] > 1)[0])
                grad_dsm_up = np.sum(dsm_mask[typ[0] - 3, min(txp):max(txp)]) / len(
                    np.where(dsm_mask[typ[0] - 3, min(txp):max(txp)] > 1)[0]) \
                              - np.sum(dsm_mask[typ[0], min(txp):max(txp)]) / len(
                    np.where(dsm_mask[typ[0], min(txp):max(txp)] > 1)[0])
                mean_ort_down = np.array(
                    [np.sum(tbd_down * ortho_mask[:, :, 0]), np.sum(tbd_down * ortho_mask[:, :, 1]), \
                     np.sum(tbd_down * ortho_mask[:, :, 2])]) / len(np.where(tbd_down * ortho_mask[:, :, 0] > 1)[0])
                grad_dsm_down = np.sum(dsm_mask[typ[0], min(txp):max(txp)]) / len(
                    np.where(dsm_mask[typ[0], min(txp):max(txp)] > 1)[0]) \
                                - np.sum(dsm_mask[typ[0] + 3, min(txp):max(txp)]) / len(
                    np.where(dsm_mask[typ[0] + 3, min(txp):max(txp)] > 1)[0])
                if (abs(np.mean(mean_ort_up - mean_ort_down)) > 5 or abs(
                        grad_dsm_up - grad_dsm_down) > 0.5) and max(np.diff(txp)) == 1:
                    sp_idxv.append(idx + min(ty))

    # direction for horizontal (from left to right)
    ort_dh = []
    dsm_dh = []
    for j in range(max(tx) - min(tx)):
        ort_dh.append(sum(ortho_mask[:, min(tx) + j, :]) / sum(bd_mask[:, min(tx) + j]))
        dsm_dh.append(sum(dsm_mask[:, min(tx) + j]) / sum(bd_mask[:, min(tx) + j]))
    if len(dsm_dh) > 2:
        ort_dhf = np.mean(abs(np.gradient(np.array(ort_dh))[0]), axis=1)
        dsm_dhf = abs(np.gradient(np.array(dsm_dh).T).T)
    else:
        ort_dhf = []
        dsm_dhf = []
    if len(ort_dhf) < 1:
        sp_idx = []
        sp_idxh = []
    else:
        sp_idxd = list(signal.find_peaks(-np.array(dsm_dh), distance=10)[0])
        sp_idxd.extend(list(np.where(dsm_dhf > 0.2)[0]))
        sp_idxor = list(signal.find_peaks(ort_dhf, height=10, distance=10)[0])
        sp_idxor1 = list(np.where(np.array(ort_dvf) > 50)[0])
        sp_idx = list((set(sp_idxor).intersection(set(sp_idxd))).union(set(sp_idxor1).intersection(set(sp_idxor))))
        sp_idxh = []
        for j in range(len(np.where(np.array(sp_idx) < 10)[0]), 0, -1):
            sp_idx.pop(j - 1)
        for j in range(len(np.where(np.array(sp_idx) > max(tx) - min(tx) - 10)[0]), 0, -1):
            sp_idx.pop(j - 1)
        if len(sp_idx) > 0:
            for j, idx in enumerate(sp_idx):
                diff_mask = copy.deepcopy(bd_mask)
                diff_mask[:, idx + min(ty)] = 0
                diff_mask = bd_mask - diff_mask
                (typ, txp) = np.where(diff_mask == 1)
                tbd_left = 0 * bd_mask
                tbd_right = 0 * bd_mask
                if len(typ) < 1:
                    continue
                tbd_left[min(typ):max(typ), txp[0] - 3:txp[0]] = 1
                tbd_right[min(typ):max(typ), txp[0] - 3:txp[0]] = 1
                mean_ort_left = np.array(
                    [np.sum(tbd_left * ortho_mask[:, :, 0]), np.sum(tbd_left * ortho_mask[:, :, 1]), \
                     np.sum(tbd_left * ortho_mask[:, :, 2])]) / len(np.where(tbd_left * ortho_mask[:, :, 0] > 1)[0])
                grad_dsm_left = np.sum(dsm_mask[min(typ):max(typ), txp[0] - 3]) / len(
                    np.where(dsm_mask[min(typ):max(typ), txp[0] - 3] > 1)[0]) \
                                - np.sum(dsm_mask[min(typ):max(typ), txp[0]]) / len(
                    np.where(dsm_mask[min(typ):max(typ), txp[0]] > 1)[0])
                mean_ort_right = np.array(
                    [np.sum(tbd_right * ortho_mask[:, :, 0]), np.sum(tbd_right * ortho_mask[:, :, 1]), \
                     np.sum(tbd_right * ortho_mask[:, :, 2])]) / len(
                    np.where(tbd_right * ortho_mask[:, :, 0] > 1)[0])
                grad_dsm_right = np.sum(dsm_mask[min(typ):max(typ), txp[0]]) / len(
                    np.where(dsm_mask[min(typ):max(typ), txp[0]] > 1)[0]) \
                                 - np.sum(dsm_mask[min(typ):max(typ), txp[0] + 3]) / len(
                    np.where(dsm_mask[min(typ):max(typ), txp[0] + 3] > 1)[0])
                if (abs(np.mean(mean_ort_left - mean_ort_right)) > 5 or abs(
                        grad_dsm_left - grad_dsm_right) > 0.5) and max(np.diff(typ)) == 1:
                    sp_idxh.append(idx + min(tx))

    # 3. generate pyramid maximum inner rectangle in each grid level
    # generate pyramid mask
    bd_mask_t = copy.deepcopy(bd_mask)
    im_4 = cv2.resize(bd_mask_t, None, fx=0.25, fy=0.25).astype(int)
    im_2 = cv2.resize(bd_mask_t, None, fx=0.5, fy=0.5).astype(int)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    im_40 = cv2.morphologyEx(im_4.astype(float), cv2.MORPH_CLOSE, kernel).astype(int)
    if len(sp_idxv) > 0:
        bd_mask_t[sp_idxv, :] = 0
        im_4[[round(k / 4) for k in sp_idxv], :] = 0
        im_40[[round(k / 4) for k in sp_idxv], :] = 0
        im_2[[round(k / 2) for k in sp_idxv], :] = 0
    if len(sp_idxh) > 0:
        bd_mask_t[:, sp_idxh] = 0
        im_4[:, [round(k / 4) for k in sp_idxh]] = 0
        im_40[:, [round(k / 4) for k in sp_idxh]] = 0
        im_2[:, [round(k / 2) for k in sp_idxh]] = 0
    im_1 = bd_mask_t.astype(int)

    # get maximum rectangles
    count_im4 = np.sum(im_4)
    (m4, n4) = np.shape(im_4)
    (m2, n2) = np.shape(im_2)
    (m1, n1) = np.shape(im_1)
    decp4 = []
    decp2 = []
    decp1 = []

    recori = []
    flagnum = 0
    min_area = 4

    # grid level 1
    while count_im4 > min_area and flagnum < 50:
        LRout = findRotMaxRect((1 - im_40), flag_opt=True, main_angle=0)
        for j, lrout in enumerate(LRout):
            lrout[0], lrout[1] = lrout[1], lrout[0]
        areaLR = (sqrt((LRout[0][0] - LRout[1][0]) ** 2 + (LRout[0][1] - LRout[1][1]) ** 2) + 1) * \
                 (sqrt((LRout[0][0] - LRout[3][0]) ** 2 + (LRout[0][1] - LRout[3][1]) ** 2) + 1)
        LRout_sd = [[0, 0], [0, 0], [0, 0], [0, 0]]
        rec = np.array(LRout).T
        LRout_list = []
        arealr = []
        for j in range(len(orie)):
            if abs(sum(rec[:, 0]) - sum(rec[:, 1])) > 1 and abs(sum(rec[:, 0]) - sum(rec[:, 3])) > 1:
                LRout_sd = findRotMaxRect((1 - im_40), flag_opt=True, main_angle=orie[j])
                for k, lrout in enumerate(LRout_sd):
                    lrout[0], lrout[1] = lrout[1], lrout[0]
            else:
                LRout_sd = [[0, 0], [0, 0], [0, 0], [0, 0]]
            arealr.append((sqrt(
                (LRout_sd[0][0] - LRout_sd[1][0] + 1) ** 2 + (LRout_sd[0][1] - LRout_sd[1][1] + 1) ** 2) + 1) * \
                          (sqrt((LRout_sd[0][0] - LRout_sd[3][0] + 1) ** 2 + (
                                      LRout_sd[0][1] - LRout_sd[3][1] + 1) ** 2) + 1))
            LRout_list.append(LRout_sd)
        if len(arealr) > 0:
            arealr_max = max(arealr)
            ori_num = arealr.index(arealr_max)
        if len(orie) < 1:
            arealr_max = 0
            ori_num = -1
        flagnum += 1
        if areaLR >= arealr_max and areaLR > min_area:
            im_40[int(LRout[0][1]):int(LRout[2][1]), int(LRout[0][0]):int(LRout[2][0])] = 0
            count_im4 -= areaLR
            if abs(sum(rec[:, 0]) - sum(rec[:, 1])) < 1 or abs(sum(rec[:, 0]) - sum(rec[:, 3])) < 1 or max(areaLR,
                                                                                                           arealr_max) < min_area:
                rec = []
            else:
                recori.append(0)
        else:
            if ori_num == -1 or areaLR < min_area:
                continue
            LRout_sd = LRout_list[ori_num]
            txy = np.array(LRout_sd)
            im_40t = draw.polygon2mask([m4, n4], np.array([txy[:, 1], txy[:, 0]]).T)
            im_40 = im_40 - im_40t
            rec = txy.T
            count_im4 -= (sqrt(
                (LRout_sd[0][0] - LRout_sd[1][0]) ** 2 + (LRout_sd[0][1] - LRout_sd[1][1]) ** 2) + 1) * \
                         (sqrt((LRout_sd[0][0] - LRout_sd[3][0]) ** 2 + (LRout_sd[0][1] - LRout_sd[3][1]) ** 2) + 1)
            recori.append(orie[ori_num])
        decp4.append(rec)
    decp4 = [x for x in decp4 if x != []]
    decp_num0 = len(decp4)
    if decp_num0 == 0:
        return(decp_rect)

    # grid level 2
    im_20 = copy.deepcopy(im_2)
    for j, decp in enumerate(decp4):
        rec = np.array(decp).T
        rec0 = copy.deepcopy(rec)
        rec0[np.where(rec0[:, 0] == min(rec0[:, 0])), 0] = min(rec0[:, 0]) - 1
        rec0[np.where(rec0[:, 1] == min(rec0[:, 1])), 1] = min(rec0[:, 1]) - 1
        bd_im4 = draw.polygon2mask([m4, n4], np.array([rec0[:, 1], rec0[:, 0]]).T).astype(float)
        (recr0, recc0) = np.where(bd_im4 == 1)
        if len(recr0) < 1:
            continue
        if min(recr0) == max(recr0):
            if rec[0, 1] != rec[2, 1]:
                bd_im4 = draw.polygon2mask([m4, n4], np.array(
                    [np.array([rec[0, 1] - 1, rec[1, 1] - 1, rec[2, 1], rec[3, 1]]).T, rec[:, 0]]).T).astype(float)
            else:
                bd_im4 = draw.polygon2mask([m4, n4], np.array(
                    [np.array([rec[0, 1] - 1, rec[1, 1], rec[2, 1] - 1, rec[3, 1]]).T, rec[:, 0]]).T).astype(float)
        bd_im4r_s2 = cv2.resize(bd_im4, (n2, m2)).astype(int)

        decp, bd_im2f = getgridmask(bd_im4r_s2, recori[j], im_20)

        decp2.append(decp)
        im_20 = im_20 - bd_im2f

    # grid level 3
    decp2 = [x for x in decp2 if x != []]
    im_10 = copy.deepcopy(im_1)
    for j, decp in enumerate(decp2):
        rec = np.array(decp).T
        rec0 = copy.deepcopy(rec)
        rec0[np.where(rec0[:, 0] == min(rec0[:, 0])), 0] = min(rec0[:, 0]) - 1
        rec0[np.where(rec0[:, 1] == min(rec0[:, 1])), 1] = min(rec0[:, 1]) - 1
        bd_im2 = draw.polygon2mask([m2, n2], np.array([rec0[:, 1], rec0[:, 0]]).T).astype(float)
        (recr0, recc0) = np.where(bd_im2 == 1)
        if len(recr0) < 1:
            continue
        if min(recr0) == max(recr0):
            if rec[0, 1] != rec[2, 1]:
                bd_im2 = draw.polygon2mask([m2, n2], np.array(
                    [np.array([rec[0, 1] - 1, rec[1, 1] - 1, rec[2, 1], rec[3, 1]]).T, rec[:, 0]]).T).astype(float)
            else:
                bd_im2 = draw.polygon2mask([m2, n2], np.array(
                    [np.array([rec[0, 1] - 1, rec[1, 1], rec[2, 1] - 1, rec[3, 1]]).T, rec[:, 0]]).T).astype(float)
        bd_im2r_s1 = cv2.resize(bd_im2, (n1, m1)).astype(int)

        decp, bd_im1f = getgridmask(bd_im2r_s1, recori[j], im_10)

        if len(decp) > 0:
            decp1.append(decp)
        im_10 = im_10 - bd_im1f

        # 4. seperation and merging
    decp1 = [x for x in decp1 if x != []]
    decp_num0 = len(decp1)
    if decp_num0 > 1:
        # seperate
        spdecp = []
        for j in range(decp_num0):
            rec = np.array(decp1[j]).T.astype(int)
            bd_im1 = draw.polygon2mask([m1, n1], np.array(
                [np.array([rec[0, 1] - 1, rec[1, 1] - 1, rec[2, 1], rec[3, 1]]).T, rec[:, 0]]).T).astype(int)
            area_bd = np.sum(bd_im1)
            if area_bd > 1000 and recori[j] == 0:
                centd = np.array([ceil((rec[0, 0] + rec[1, 0]) / 2), ceil((rec[1, 1] + rec[2, 1]) / 2)])
                newrec = []
                if abs(rec[0, 0] - rec[1, 0]) > abs(rec[1, 1] - rec[2, 1]):
                    grad_dsm = np.gradient(dsm_mask[ceil((rec[1, 1] + rec[2, 1]) / 2), rec[0, 0]:rec[1, 0]])
                    sp_idx = list(signal.find_peaks(abs(grad_dsm), height=0.25, distance=10)[0])
                    for q in range(len(np.where(np.array(sp_idx) < 10)[0]), 0, -1):
                        sp_idx.pop(q - 1)
                    for q in range(len(np.where(np.array(sp_idx) > len(grad_dsm) - 10)[0]), 0, -1):
                        sp_idx.pop(q - 1)
                    sp_idx0 = []
                    if len(sp_idx) > 0:
                        for k in range(len(sp_idx)):
                            ts_mask = ortho_mask[centd[1] - 10:centd[1] + 10,
                                      rec[0, 0] + sp_idx[k] - 10:rec[0, 0] + sp_idx[k] + 10, :]
                            tsr, tsg, tsb = ts_mask[:, :, 0], ts_mask[:, :, 1], ts_mask[:, :, 2]
                            avg_std_rgb = np.mean(np.array([np.std(tsr[np.where(tsr > 1)]), \
                                                            np.std(tsg[np.where(tsg > 1)]),
                                                            np.std(tsb[np.where(tsb > 1)])]))
                            flagsp = 1
                            if len(sp_idx0) > 0:
                                flagsp = sp_idx0[len(sp_idx0) - 1]
                            if avg_std_rgb > 12 and abs(flagsp - sp_idx[k]) > 10:
                                sp_idx0.append(sp_idx[k])
                        prex = rec[0, 0]
                        sp_idx0.append(len(grad_dsm))
                        if len(sp_idx0) > 1:
                            for k in range(len(sp_idx0)):
                                newrec.append(
                                    np.array([[prex, rec[0, 0] + sp_idx0[k], rec[0, 0] + sp_idx0[k], prex], \
                                              [rec[0, 1], rec[1, 1], rec[2, 1], rec[3, 1]]]))
                                prex = rec[0, 0] + sp_idx0[k]
                        else:
                            newrec.append(decp1[j])
                        # newrec=np.array(newrec)
                    else:
                        newrec.append(decp1[j])
                else:
                    grad_dsm = np.gradient(dsm_mask[rec[1, 1]:rec[2, 1], ceil((rec[0, 0] + rec[1, 0]) / 2)])
                    sp_idx = list(signal.find_peaks(abs(grad_dsm), height=0.25, distance=10)[0])
                    for q in range(len(np.where(np.array(sp_idx) < 10)[0]), 0, -1):
                        sp_idx.pop(q - 1)
                    for q in range(len(np.where(np.array(sp_idx) > len(grad_dsm) - 10)[0]), 0, -1):
                        sp_idx.pop(q - 1)
                    sp_idx0 = []
                    if len(sp_idx) > 0:
                        for k in range(len(sp_idx)):
                            ts_mask = ortho_mask[rec[0, 1] + sp_idx[k] - 10:rec[0, 1] + sp_idx[k] + 10,
                                      centd[1] - 10:centd[1] + 10, :]
                            tsr, tsg, tsb = ts_mask[:, :, 0], ts_mask[:, :, 1], ts_mask[:, :, 2]
                            avg_std_rgb = np.mean(np.array([np.std(tsr[np.where(tsr > 1)]), \
                                                            np.std(tsg[np.where(tsg > 1)]),
                                                            np.std(tsb[np.where(tsb > 1)])]))
                            flagsp = 1
                            if len(sp_idx0) > 0:
                                flagsp = sp_idx0[len(sp_idx0) - 1]
                            if avg_std_rgb > 12 and abs(flagsp - sp_idx[k]) > 10:
                                sp_idx0.append(sp_idx[k])
                        prey = rec[0, 1]
                        sp_idx0.append(len(grad_dsm))
                        if len(sp_idx0) > 1:
                            for k in range(len(sp_idx0)):
                                newrec.append(np.array([[rec[0, 0], rec[1, 0], rec[2, 0], rec[3, 0]], \
                                                        [prey, prey, rec[0, 1] + sp_idx0[k],
                                                         rec[0, 1] + sp_idx0[k]]]))
                                prey = rec[0, 1] + sp_idx0[k]
                        else:
                            newrec.append(decp1[j])
                        # newrec=np.array(newrec)
                    else:
                        newrec.append(decp1[j])
                spdecp.extend(newrec)
            else:
                spdecp.append(decp1[j])

        # merge
        sp_num = len(spdecp)
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mean_rgb = np.zeros((sp_num, 3), dtype=float)
        std_rgb = np.zeros((sp_num, 3), dtype=float)
        mean_dsm = np.zeros((sp_num, 1), dtype=float)
        cent_bd = np.zeros((sp_num, 2), dtype=float)
        area_bd = np.zeros((sp_num, 1), dtype=float)
        mgdecp = []
        for j in range(sp_num):
            rec = np.array(spdecp[j]).T
            bd_im1 = draw.polygon2mask([m1, n1], np.array(
                [np.array([rec[0, 1] - 1, rec[1, 1] - 1, rec[2, 1], rec[3, 1]]).T, rec[:, 0]]).T).astype(int)
            area_bd[j] = np.sum(bd_im1)
            mask_r = bd_im1 * ortho_mask[:, :, 1]
            mask_g = bd_im1 * ortho_mask[:, :, 1]
            mask_b = bd_im1 * ortho_mask[:, :, 1]
            mean_rgb[j, :] = np.array([mask_r.sum(), mask_g.sum(), mask_b.sum()]) / len(np.where(mask_r > 1)[0])
            std_rgb[j, :] = np.array([np.std(mask_r[np.where(mask_r > 1)]), np.std(mask_g[np.where(mask_g > 1)]), \
                                      np.std(mask_b[np.where(mask_b > 1)])])
            mean_dsm[j] = np.sum(bd_im1 * dsm_mask) / area_bd[j]
            cent_bd[j, :] = np.array([ceil((rec[0, 0] + rec[1, 0]) / 2), ceil((rec[1, 1] + rec[2, 1]) / 2)])
        flagmat = np.triu(np.ones((sp_num, sp_num), dtype=int)) + np.diag(np.ones((sp_num, sp_num), dtype=int))
        for j in range(sp_num):
            rec = np.array(spdecp[j]).T.astype(int)
            rec0 = copy.deepcopy(rec)
            rec0[np.where(rec0[:, 0] == min(rec0[:, 0])), 0] = min(rec0[:, 0]) - 1
            rec0[np.where(rec0[:, 1] == min(rec0[:, 1])), 1] = min(rec0[:, 1]) - 1
            bd_im1 = draw.polygon2mask([m1, n1], np.array([rec0[:, 1], rec0[:, 0]]).T).astype(int)
            inlrec = cv2.dilate(bd_im1.astype(float), kernel_m, iterations=1).astype(int)
            for k in range(sp_num):
                if flagmat[j, k] != 1:
                    continue
                rec1 = np.array(spdecp[k]).T
                rec0 = copy.deepcopy(rec1)
                rec0[np.where(rec0[:, 0] == min(rec0[:, 0])), 0] = min(rec0[:, 0]) - 1
                rec0[np.where(rec0[:, 1] == min(rec0[:, 1])), 1] = min(rec0[:, 1]) - 1
                bd_im1cp = draw.polygon2mask([m1, n1], np.array([rec0[:, 1], rec0[:, 0]]).T).astype(int)
                inlreccp = cv2.dilate(bd_im1cp.astype(float), kernel_m, iterations=1).astype(int)
                dm_rgb = np.mean(abs(mean_rgb[j, :] - mean_rgb[k, :]))
                ds_rgb = np.mean(abs(std_rgb[j, :] - std_rgb[k, :]))
                dm_dsm = abs(mean_dsm[j] - mean_dsm[k])
                dxy = cent_bd[j, :] - cent_bd[k, :]
                bd_int = cv2.dilate(inlrec * inlreccp.astype(float), kernel_m, iterations=1).astype(int)
                mask_r, mask_g, mask_b = bd_int * ortho_mask[:, :, 0], bd_int * ortho_mask[:, :,
                                                                                1], bd_int * ortho_mask[:, :, 2]
                mean_rgb_int = np.array([mask_r.sum(), mask_g.sum(), mask_b.sum()]) / len(np.where(mask_r > 1)[0])
                std_rgb_int = np.array([np.std(mask_r[np.where(mask_r > 1)]), np.std(mask_g[np.where(mask_g > 1)]), \
                                        np.std(mask_b[np.where(mask_b > 1)])])
                mean_dsm_int = np.sum(bd_int * dsm_mask) / np.sum(bd_int)
                dm_int = abs(mean_dsm_int - (mean_dsm[j] - mean_dsm[k]))
                itmrgb = abs(np.mean(mean_rgb_int - abs(mean_rgb[j, :] + mean_rgb[k, :]) / 2))
                itsrgb = abs(np.mean(std_rgb_int - abs(std_rgb[j, :] + std_rgb[k, :]) / 2))
                if itmrgb < Td and itsrgb < 20 and dm_int < Th1:
                    tagmerge = (dm_rgb < 5 and ds_rgb < 15 and dm_dsm < Th1 and min(abs(dxy)) < 5) \
                               or (((min(itmrgb, itsrgb) < Td and max(itmrgb, itsrgb) < 8) or (
                                itmrgb < Td / 2 and itsrgb < 10)) \
                                   and dm_int < Th2 * 2 and min(abs(dxy)) < 5)
                    if tagmerge == 1:
                        if abs(dxy[0]) > abs(dxy[1]):
                            dy = floor((cent_bd[j, 1] * area_bd[j] + cent_bd[k, 1] * area_bd[k]) / (
                                        area_bd[j] + area_bd[k]))
                            recxy1 = min(min(rec[:, 0]), min(rec1[:, 0]))
                            recxy2 = max(max(rec[:, 0]), max(rec1[:, 0]))
                            recxy3 = np.mean(np.array(
                                [min(rec[:, 1]) - (cent_bd[j, 1] - dy), min(rec1[:, 1]) - (cent_bd[k, 1] - dy)]))
                            recxy4 = np.mean(np.array(
                                [max(rec[:, 1]) - (cent_bd[j, 1] - dy), max(rec1[:, 1]) - (cent_bd[k, 1] - dy)]))
                            mgdecp.append(
                                np.array([[recxy1, recxy2, recxy2, recxy1], [recxy3, recxy3, recxy4, recxy4]]))
                        else:
                            dx = floor((cent_bd[j, 0] * area_bd[j] + cent_bd[k, 0] * area_bd[k]) / (
                                        area_bd[j] + area_bd[k]))
                            recxy1 = np.mean(np.array(
                                [min(rec[:, 0]) - (cent_bd[j, 0] - dx), min(rec1[:, 0]) - (cent_bd[k, 0] - dx)]))
                            recxy2 = np.mean(np.array(
                                [max(rec[:, 0]) - (cent_bd[j, 0] - dx), max(rec1[:, 0]) - (cent_bd[k, 0] - dx)]))
                            recxy3 = min(min(rec[:, 1]), min(rec1[:, 1]))
                            recxy4 = max(max(rec[:, 1]), max(rec1[:, 1]))
                            mgdecp.append(
                                np.array([[recxy1, recxy2, recxy2, recxy1], [recxy3, recxy3, recxy4, recxy4]]))
                        flagmat[j, :] = 0
                        flagmat[:, k] = 0
                        flagmat[k, :] = 0
            if max(flagmat[j, :]) == 2:
                mgdecp.append(spdecp[j])
    else:
        mgdecp = copy.deepcopy(decp1)

    # 5. transfer coordinate to whole image
    jflag = 1
    mgdecp_num = len(mgdecp)
    decp_rec = []
    for j in range(mgdecp_num):
        recp = np.array(mgdecp[j]).T + np.array([min(pt_trans[:, 0]) - 10, min(pt_trans[:, 1]) - 10])
        pr_trans0 = np.dot(r1, recp.T).T
        pr_trans0[:, 0] += pt_cent[1]
        pr_trans0[:, 1] += pt_cent[0]
        pr_trans = copy.deepcopy(pr_trans0)
        pr_trans[:, 0] = pr_trans0[:, 0] + range_local[2]
        pr_trans[:, 1] = pr_trans0[:, 1] + range_local[0]
        tx = np.array([round(k) for k in pr_trans0[:, 0]])
        ty = np.array([round(k) for k in pr_trans0[:, 1]])
        tx[np.where(tx < 1)] = 1
        ty[np.where(ty < 1)] = 1
        mask1 = np.zeros((mask_loc[0], mask_loc[1]), dtype=int)
        mask0 = draw.polygon2mask([mask_loc[0], mask_loc[1]], np.array([ty, tx]).T).astype(int)
        area0 = np.sum(mask0)
        mask1[min(ty):max(ty), min(tx):max(tx)] = img_mask[min(ty) + range_local[0]:max(ty) + range_local[0], \
                                                  min(tx) + range_local[2]:max(tx) + range_local[2]]
        over2d = np.sum(mask1 * mask0) / area0
        areadecp = (mgdecp[j][0, 2] - mgdecp[j][0, 0]) * (mgdecp[j][1, 2] - mgdecp[j][1, 0])
        # if over2d>0.7:
        decp_rec.append([i, jflag, pr_trans[0, 0], pr_trans[0, 1], pr_trans[1, 0], pr_trans[1, 1] \
                            , pr_trans[3, 0], pr_trans[3, 1], pr_trans[2, 0], pr_trans[2, 1], int(areadecp),
                         sita_v])
        jflag += 1
    decp_rect=copy.deepcopy(decp_rec)

    return decp_rect