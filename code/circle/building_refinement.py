from skimage import io,data,filters,segmentation,measure,morphology,feature,draw
from skimage.util import img_as_ubyte
from scipy import ndimage
import numpy as np
import shapefile
from math import pi,sqrt,sin,cos,exp
import cv2
import copy
import json
from building_modelfit import fitrec



#construct 2D shape format for each rectangle
def construct2d(decp_rec):
    shape2d=[]
    
    for i in range(len(decp_rec)):
        for j, rec in enumerate(decp_rec[i]):
            x0=np.mean(rec[2:9:2])
            y0=np.mean(rec[3:10:2])
            if len(rec)<1:
                continue
            length0=sqrt((rec[2]-rec[4])**2+(rec[3]-rec[5])**2)
            width0=sqrt((rec[2]-rec[6])**2+(rec[3]-rec[7])**2)
            orient0=np.arctan((rec[3]-rec[5])/(rec[2]-rec[4]))
            para2d=np.array([x0,y0,length0,width0,orient0,0])
            alpha=np.arctan(para2d[3]/para2d[2])
            beta=pi/2-para2d[4]-alpha
            x1=para2d[0]+sin(beta)*sqrt(para2d[2]**2+para2d[3]**2)/2
            y1=para2d[1]+cos(beta)*sqrt(para2d[2]**2+para2d[3]**2)/2
            x2=x1-para2d[2]*cos(para2d[4])
            y2=y1-para2d[2]*sin(para2d[4])
            x3=x1+para2d[3]*sin(para2d[4])
            y3=y1-para2d[3]*cos(para2d[4])
            x4=x2+para2d[3]*sin(para2d[4])
            y4=y2-para2d[3]*cos(para2d[4])
            tempnp=np.array([i,j,x0,y0,length0,width0,orient0,0,x1,y1,x2,y2,x3,y3,x4,y4])
            if np.isnan(np.min(tempnp)):
                continue
            shape2d.append(np.array([i,j,x0,y0,length0,width0,orient0,0,x1,y1,x2,y2,x3,y3,x4,y4]))
    
    return shape2d


def process_shp(shp,tfw,imgsize):
    line_ori=[]
    area_street=[]
    pix=tfw[0]*100
    for sp in shp:
        line_len=len(sp.points)
        line_x1, line_x2=sp.points[0][0],sp.points[line_len-1][0]
        line_y1, line_y2=sp.points[0][1],sp.points[line_len-1][1]
        lx, ly=(line_x1+line_x2)/2, (line_y1+line_y2)/2
        if lx>tfw[4]-pix and lx<tfw[4]+tfw[0]*imgsize[1]+pix and ly<tfw[5]+pix and ly>tfw[5]+tfw[3]*imgsize[0]-pix:
            if line_x2==line_x1:
                line_ori.append(pi/2)
            else:
                line_ori.append(-np.arctan((line_y2-line_y1)/(line_x2-line_x1)))
            area_street.append(sp)
    
    st_line=[]
    for st_temp in area_street:
        for j in range(len(st_temp.points)-2):
            line_x1, line_x2=st_temp.points[j][0], st_temp.points[j+1][0]
            line_y1, line_y2=st_temp.points[j][1], st_temp.points[j+1][1]
            if line_x2==line_x1:
                ori=pi/2
            else:
                ori=-np.arctan((line_y2-line_y1)/(line_x2-line_x1))
            st_line.append(np.array([(line_x1+line_x2)/2,(line_y1+line_y2)/2,ori]))
    
    return line_ori, st_line


def CalculateAffinity(datamat):
    sigma=1
    affinity=np.zeros((len(datamat),len(datamat)),dtype=float)
    for i, datai in enumerate(datamat):
        for j, dataj in enumerate(datamat):
            dist=np.sqrt(np.sum((datai-dataj)**2))
            affinity[i,j]=exp(-dist/(2*sigma**2))
    return affinity


def extract_img_geo(tfw,imgsize):
    img_geo_x=[]
    img_geo_y=[]
    for i in range(imgsize[0]):
        img_geo_y.append(i*tfw[3]+tfw[5])
    for i in range(imgsize[1]):
        img_geo_x.append(i*tfw[0]+tfw[4])
    return img_geo_x,img_geo_y

#Use OpenStreetMap line segment to refine the orientation
def construct2d_OSM(shape2d,img_geo_x,img_geo_y,st_line):
    shape2d_osm=[]
    for i, rec in enumerate(shape2d):
        x0=np.mean(rec[8::2])
        y0=np.mean(rec[9::2])
        length0, width0 = rec[4], rec[5]
        ori0=rec[6]
        ori_edit=ori0
        x0s, y0s=img_geo_x[max(round(x0)-100,0)], img_geo_y[min(round(y0)+100,len(img_geo_y)-100)]
        dist=[]
        
        for stl in st_line:
            line_center=stl[0:2]
            dist.append((x0s-line_center[0])**2+(y0s-line_center[1])**2)
        
        pos=np.argsort(np.array(sorted(range(len(dist)), key=lambda k: dist[k], reverse=False)))
        ori_street=np.zeros((3,1),dtype=float)
        if len(pos)>0:
            ori_street=np.array(st_line)[pos[0:3]-1,2]


        ang_thr=pi/18
        for ori_n in range(3):
            if ori_street[ori_n]<0:
                ori_street[ori_n]+=pi/2
        if max(ori_street)-min(ori_street)<pi/36:
            ori_m=np.mean(ori_street)
            if abs(ori0-ori_m)<ang_thr:
                ori_edit=ori_m
            elif abs(ori0-ori_m+pi/2)<ang_thr:
                ori_edit = ori_m - pi / 2
            elif abs(ori0 - ori_m - pi/2) < ang_thr:
                ori_edit = ori_m - pi / 2
        else:
            ori_diff=np.array([ori_street[0]-ori_street[1],ori_street[1]-ori_street[2],ori_street[2]-ori_street[0]])
            ori_m_array=np.array([(ori_street[0]+ori_street[1])/2,(ori_street[1]+ori_street[2])/2,\
                                (ori_street[2]+ori_street[0])/2])
            ori_flag=np.argmin(ori_diff)
            ori_m=ori_m_array[ori_flag]
            if abs(ori_edit-ori_m)<ang_thr:
                ori_edit=ori_m
            elif abs(ori0-ori_m+pi/2)<ang_thr:
                ori_edit=ori_m-pi/2
            elif abs(ori0 - ori_m - pi/2) < ang_thr:
                ori_edit = ori_m - pi / 2
        
        para2d=np.array([x0,y0,length0,width0,ori_edit,0])
        alpha=np.arctan(para2d[3]/para2d[2])
        beta=pi/2-para2d[4]-alpha
        x1=para2d[0]+sin(beta)*sqrt(para2d[2]**2+para2d[3]**2)/2
        y1=para2d[1]+cos(beta)*sqrt(para2d[2]**2+para2d[3]**2)/2
        x2=x1-para2d[2]*cos(para2d[4])
        y2=y1-para2d[2]*sin(para2d[4])
        x3=x1+para2d[3]*sin(para2d[4])
        y3=y1-para2d[3]*cos(para2d[4])
        x4=x2+para2d[3]*sin(para2d[4])
        y4=y2-para2d[3]*cos(para2d[4])    
        shape2d_osm.append(np.array([rec[0],rec[1],x0,y0,length0,width0,ori_edit,0,x1,y1,x2,y2,x3,y3,x4,y4]))
        
    return shape2d_osm


#Directly derive mesh from DSM for irregular building shape
def derive_irreg(decp_rec,pt_list_int,L_mask):
    decp_rec_reg=copy.deepcopy(decp_rec)
    decp_ir=[]
    for i in range(len(decp_rec)):
        rec_area = 0
        imgsize = np.shape(L_mask)
        img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        rec_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        img_mask[np.where(L_mask == i+1)] = 1
        mask_area = np.sum(img_mask)
        for j, rec in enumerate(decp_rec[i]):
            if len(rec) < 1:
                continue
            rec_area += rec[10]
            tx, ty = np.rint(rec[2:9:2]), np.rint(rec[3:10:2])
            tx[np.where(tx > imgsize[1] - 1)] = imgsize[1] - 1
            ty[np.where(ty > imgsize[0] - 1)] = imgsize[0] - 1
            mask_r = fitrec(imgsize[0], imgsize[1], tx, ty)
            rec_mask+=mask_r.astype(np.uint8)
        intersection = np.logical_and(img_mask, rec_mask)
        union = np.logical_or(img_mask, rec_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        if iou_score < 0.65 and mask_area>5000:
            decp_rec_reg[i] = []
            decp_ir.append(pt_list_int[i])
    return decp_rec_reg,decp_ir
    