from skimage import io,data,filters,segmentation,measure,morphology,color,feature,draw
from skimage.util import img_as_ubyte
from scipy import ndimage
from rdp import rdp
import numpy as np
from math import pi,sqrt,sin,cos
import cv2
import copy
import json

def initial_line_extraction(label_mask,label_num):
    imgsize=np.shape(label_mask)
    dp_pt=[]
    dp_line=[]
    for i in range(1,label_num+1):
        img_mask=np.zeros((imgsize[0],imgsize[1]),dtype=np.uint8)
        img_mask[np.where(label_mask==i)]=1
        #Douglas-Peucker algorithm
        contours=measure.find_contours(img_mask, 0.5)
        dp_pt=rdp(contours[0],epsilon=3, algo="iter", return_mask=False)
        temp_line=[]
        for j in range(len(dp_pt)-1):
            pt1=np.array(dp_pt[j])
            pt2=np.array(dp_pt[j+1])
            pt1[0],pt1[1]=pt1[1],pt1[0]
            pt2[0],pt2[1]=pt2[1],pt2[0]
            angle_line=np.arctan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
            temp_line.append([angle_line,pt1,pt2,j])
        dp_line.append(temp_line)
    return dp_line

def line_adjustment(dp_line, label_num):
    adj_line=[]
    #find main orientation
    for i in range(label_num):
        poly1_line=[]
        line_ori=[]
        for j, line_p in enumerate(dp_line[i]):
            temp_line=np.array(line_p)
            line_loc=np.array([(temp_line[1][0]+temp_line[2][0])/2,\
                               (temp_line[1][1]+temp_line[2][1])/2])
            line_len=sqrt((temp_line[2][0]-temp_line[1][0])**2\
                             +(temp_line[2][1]-temp_line[1][1])**2)
            poly1_line.append([temp_line[0],line_loc,line_len,temp_line[1][:],temp_line[2][:],temp_line[3]])
        poly2_line=[]
        flag=1
        #adjust orienation into 9 groups
        while (len(poly2_line)<len(dp_line[i])):
            angle_thr=pi/18
            angle_line_num=np.zeros((9,1),dtype=np.float64)
            line_ori=[]
            for k, polyline in enumerate(poly1_line):
                line_ori.append(polyline[0])
            line_ori=np.array(line_ori)
            for angle_num in range(9):
                #group parallel and vetical
                angle1=-pi/2+angle_thr*(angle_num+1)
                angle2=angle1+pi/2
                angle_idx1=np.where(line_ori<=angle1)
                angle_idx2=np.where(line_ori>angle1-angle_thr)
                angle_idx3=np.where(line_ori<=angle2)
                angle_idx4=np.where(line_ori>angle2-angle_thr)
                angle_idx=set(angle_idx1[0])&set(angle_idx2[0])|set(angle_idx3[0])&set(angle_idx4[0])
                if len(angle_idx)==0:
                    angle_line_num[angle_num]=0
                elif len(angle_idx)==1:
                    temp_line=poly1_line[list(angle_idx)[0]]
                    angle_line_num[angle_num]=np.array(temp_line)[2]
                else:
                    temp_line=list(poly1_line[k] for k in list(angle_idx))
                    angle_line_num[angle_num]=sum(np.array(temp_line)[:,2])
            #from first main orienation for second iteration
            max_angle_idx=np.argmax(angle_line_num)    
            angle1=-pi/2+angle_thr*(max_angle_idx+1)
            angle2=angle1+pi/2
            angle_idxt1=set(np.where(line_ori<angle1+angle_thr/2)[0])\
                &set(np.where(line_ori>angle1-angle_thr*3/2)[0])
            angle_idxt2=set(np.where(line_ori<angle2+angle_thr/2)[0])\
                &set(np.where(line_ori>angle2-angle_thr*3/2)[0])
            temp_line_P=np.array(list(poly1_line[k] for k in list(angle_idxt1)))
            temp_line_V=np.array(list(poly1_line[k] for k in list(angle_idxt2)))
            if len(poly2_line)==len(dp_line[i]):
                break
            elif flag>=20:
                break
            #calculate weighted orientation
            angle_line=(sum([k[0] for k in temp_line_P]*np.square([k[2] for k in temp_line_P]))\
                        +sum((np.array([k[0] for k in temp_line_V])-pi/2)*np.square([k[2] for k in temp_line_V])))\
                       /(sum(np.square([k[2] for k in temp_line_P]))+sum(np.square([k[2] for k in temp_line_V])))
            angle_idxt10=[]
            angle_idxt20=[]
            if angle_line-angle_thr<-pi/2:
                angle_idxt10=np.where(line_ori>-angle_line-angle_thr)
            elif angle_line+angle_thr>0:
                angle_idxt20=np.where(line_ori<angle_line+angle_thr-pi/2)
            angle_idxt1=set(np.where(line_ori<angle_line+angle_thr)[0])\
                &set(np.where(line_ori>angle_line-angle_thr)[0])
            angle_idxt2=set(np.where(line_ori<angle_line+angle_thr+pi/2)[0])\
                &set(np.where(line_ori>angle_line-angle_thr+pi/2)[0])
            if len(angle_idxt10)>0:
                angle_idxt1=angle_idxt1|set(angle_idxt10[0])
            else:
                angle_idxt1=angle_idxt1
            if len(angle_idxt20)>0:
                angle_idxt2=angle_idxt2|set(angle_idxt20[0])
            else:
                angle_idxt2=angle_idxt2
            temp_line_P=np.array(list(poly1_line[k] for k in list(angle_idxt1)))
            temp_line_V=np.array(list(poly1_line[k] for k in list(angle_idxt2)))
            line_h1=temp_line_P
            line_h2=temp_line_V
            for k, lineh in enumerate(line_h1):
                lineh[0]=angle_line
                poly2_line.append(lineh)
            for k, lineh in enumerate(line_h2):
                lineh[0]=angle_line+pi/2
                poly2_line.append(lineh)
            for k in sorted(list(angle_idxt1|angle_idxt2))[::-1]:
                poly1_line.pop(k)
            flag+=1
        #write as line format
        
        adj_line_list=[]
        for j, polyline in enumerate(poly2_line):
            adj_line_t=np.zeros((7,1),dtype=np.float64)
            adj_line_t[0]=polyline[1][0]-polyline[2]*cos(polyline[0])/2
            adj_line_t[2]=polyline[1][1]-polyline[2]*sin(polyline[0])/2
            adj_line_t[1]=polyline[1][0]+polyline[2]*cos(polyline[0])/2
            adj_line_t[3]=polyline[1][1]+polyline[2]*sin(polyline[0])/2
            adj_line_t[4]=polyline[5]
            adj_line_t[5]=polyline[0]
            adj_line_t[6]=polyline[2]
            adj_line_t=[float(k[0]) for k in adj_line_t]
            if np.square(adj_line_t[0]-polyline[3][0])+np.square(adj_line_t[2]-polyline[3][1])>\
                np.square(adj_line_t[0]-polyline[4][0])+np.square(adj_line_t[2]-polyline[4][1]):
                adj_line_t[0],adj_line_t[1]=adj_line_t[1],adj_line_t[0]
                adj_line_t[2],adj_line_t[3]=adj_line_t[3],adj_line_t[2]
            adj_line_list.append(list(adj_line_t))
        adj_line.append(sorted(adj_line_list, key = lambda line: line[4]))
    return adj_line     

def line_regularization(adj_line, label_num, line_reg_thr, img_ortho_t):
    angle_thr=pi/9
    line_short_thr=6
    angle_reg=pi/6
    main_ori=np.zeros((label_num,1),dtype=np.float64)
    pt0=[]
    reg_line=[]
    
    #LSD detection and preprocessing
    img_ortho_gray = cv2.cvtColor(img_ortho_t, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    dlines_ortho = lsd.detect(img_ortho_gray)
    dlines_ortho = np.reshape(dlines_ortho[0],(len(dlines_ortho[0]),4))
    dlines_ortho_ct=np.zeros((len(dlines_ortho),2),dtype=np.float32)
    for i_dline in range(len(dlines_ortho)):
        dlines_ortho_ct[i_dline,0]=(dlines_ortho[i_dline,0]+dlines_ortho[i_dline,2])/2
        dlines_ortho_ct[i_dline,1]=(dlines_ortho[i_dline,1]+dlines_ortho[i_dline,3])/2
        
    line_range=20
    #label_num
    for i in range(label_num):
        #1.merge lines with same direction
        reg_line_s1=[]
        flag=-1
        for j, adjline in enumerate(adj_line[i]):
            temp_line=list(adjline)
            if j==0:
                reg_line_s1.append(temp_line)
                flag+=1
            elif (j>0)and(temp_line[5]==reg_line_s1[flag][5]):
                reg_line_s1[flag]=merge_line_function_dir(temp_line,reg_line_s1,flag)
            else:
                reg_line_s1.append(temp_line)
                flag+=1
        #merge first line and last line if directions are same
        if reg_line_s1[flag][5]==reg_line_s1[0][5]:
            temp_line=reg_line_s1[flag]
            reg_line_s1[0]=merge_line_function_dir(temp_line,reg_line_s1,0)
            reg_line_s1.pop(flag)
        
        #2.remove short lines
        reg_line_s2=[]
        reg_line_s2.append(reg_line_s1[0])
        flag=0
        if np.mean(np.array(reg_line_s1)[:,6])<30:
            line_short_thr0=4
        else:
            line_short_thr0=6
        for j in range(1,len(reg_line_s1)-1):
            if reg_line_s1[j][6]<min(line_short_thr,line_short_thr0):
                reg_line_s2[flag]=merge_line_function(reg_line_s1,reg_line_s2,j,flag)
            else:
                reg_line_s2.append(reg_line_s1[j])
                flag+=1
        if len(reg_line_s1)>2:
            if reg_line_s1[0][6]<min(line_short_thr,line_short_thr0):
                reg_line_s2[0]=merge_line_function(reg_line_s1,reg_line_s2,j+1,0)
            else:
                reg_line_s2.append(reg_line_s1[j+1])
        
        #3.adjust capable orienation of lines to main direction
        reg_line_s3=[]
        reg_line_s3.append(reg_line_s2[0])
        flag=0
        for j in range(1,len(reg_line_s2)-1):
            if (abs(reg_line_s3[flag][5]-reg_line_s2[j][5])<angle_thr) or\
                (abs(reg_line_s3[flag][5]-reg_line_s2[j][5])>pi-angle_thr):
                reg_line_s3[flag]=merge_line_function(reg_line_s2,reg_line_s3,j,flag)
            else:
                reg_line_s3.append(reg_line_s2[j])
                flag+=1
        if len(reg_line_s2)>2:
            if (abs(reg_line_s3[0][5]-reg_line_s2[j+1][5])<angle_thr) or\
                (abs(reg_line_s3[0][5]-reg_line_s2[j+1][5])>pi-angle_thr):
                tx=[reg_line_s2[j+1][0],reg_line_s3[0][1]]
                ty=[reg_line_s2[j+1][2],reg_line_s3[0][3]]
                if tx[1]==tx[0]:
                    if ty[1]>ty[0]:
                        ori=pi/2
                    else:
                        ori=-pi/2
                else:
                    ori=np.arctan((ty[1]-ty[0])/(tx[1]-tx[0]))
                reg_line_s3[0]=[tx[0],tx[1],ty[0],ty[1],float(reg_line_s3[0][4]),ori,sqrt((tx[0]-tx[1])**2+(ty[0]-ty[1])**2)]
                #reg_line_s3[0]=merge_line_function(reg_line_s3,reg_line_s2,0,j+1)
            else:
                reg_line_s3.append(reg_line_s2[j+1])
        
        #4.regularize lines to main orientation
        ori_line=np.reshape(np.array(reg_line_s3),(len(reg_line_s3),7))
        main_dir=reg_line_s3[np.argmax(ori_line[:,6])][5]
        main_dir_t=main_dir
        if main_dir_t<0:
            main_dir_t=main_dir_t+pi/2
        else:
            main_dir_t=main_dir_t-pi/2
        reg_line_s4=copy.deepcopy(reg_line_s3)
        for j, reglines3 in enumerate(reg_line_s3):
            if reglines3[6]<line_reg_thr and \
                (abs(reglines3[5]-main_dir)<angle_reg or abs(reglines3[5]-main_dir)>pi-angle_reg):
                reg_line_s4[j][5]=main_dir
            elif reglines3[6]<line_reg_thr and \
                (abs(reglines3[5]-main_dir_t)<angle_reg or abs(reglines3[5]-main_dir_t)>pi-angle_reg):
                reg_line_s4[j][5]=main_dir_t
            reg_line_s4[j][0]=sum(reglines3[0:2])/2-reglines3[6]*cos(reg_line_s4[j][5])/2
            reg_line_s4[j][1]=sum(reglines3[0:2])/2+reglines3[6]*cos(reg_line_s4[j][5])/2
            reg_line_s4[j][2]=sum(reglines3[2:4])/2-reglines3[6]*sin(reg_line_s4[j][5])/2
            reg_line_s4[j][3]=sum(reglines3[2:4])/2+reglines3[6]*sin(reg_line_s4[j][5])/2
            
            if abs(reg_line_s4[j][0]-reglines3[0])>abs(reg_line_s4[j][1]-reglines3[0]):
                reg_line_s4[j][0],reg_line_s4[j][1]=reg_line_s4[j][1],reg_line_s4[j][0]
            if abs(reg_line_s4[j][2]-reglines3[2])>abs(reg_line_s4[j][3]-reglines3[2]):
                reg_line_s4[j][2],reg_line_s4[j][3]=reg_line_s4[j][3],reg_line_s4[j][2]
            if reg_line_s4[j][5]*reglines3[5]<0:
                if abs(reglines3[0]-reglines3[1])<abs(reglines3[2]-reglines3[3]):
                    reg_line_s4[j][0],reg_line_s4[j][1]=reg_line_s4[j][1],reg_line_s4[j][0]
                else:
                    reg_line_s4[j][2],reg_line_s4[j][3]=reg_line_s4[j][3],reg_line_s4[j][2]
            
        #merge line with same direction
        reg_line_s4t=[]
        flag=-1
        for j, reglines4 in enumerate(reg_line_s4):
            temp_line=reglines4
            if j==0:
                reg_line_s4t.append(temp_line)
                flag+=1
            elif (j>0)and(temp_line[5]==reg_line_s4t[flag][5]):
                reg_line_s4t[flag]=merge_line_function_dir(temp_line,reg_line_s4t,flag)
            else:
                reg_line_s4t.append(temp_line)
                flag+=1
        if reg_line_s4t[flag][5]==reg_line_s4t[0][5]:
            temp_line=reg_line_s4t[flag]
            reg_line_s4t[0]=merge_line_function_dir(temp_line,reg_line_s4t,0)
            reg_line_s4t.pop(flag)
        
        #5.LSD orientation
        reg_line_mat=np.mat(np.reshape(np.array(copy.deepcopy(reg_line_s1)),(len(reg_line_s1),7)))
        range_x=np.array(reg_line_mat[:,0:2]).flatten()
        range_y=np.array(reg_line_mat[:,2:4]).flatten()
        range_loc=np.array([min(range_y),max(range_y),min(range_x),max(range_x)])
        
        edge_idx=(set(np.where(dlines_ortho_ct[:,0]>range_loc[2]-line_range)[0])&\
            set(np.where(dlines_ortho_ct[:,0]<range_loc[3]+line_range)[0]))&\
            (set(np.where(dlines_ortho_ct[:,1]>range_loc[0]-line_range)[0])&\
            set(np.where(dlines_ortho_ct[:,1]<range_loc[1]+line_range)[0]))
        edge_cr=[]
        edge_cr.extend([dlines_ortho[k,:] for k in edge_idx])
        edge_cr=np.array(edge_cr)
        if len(edge_cr)>1:
            edge_sita=-np.arctan((edge_cr[:,2]-edge_cr[:,0])/(edge_cr[:,3]-edge_cr[:,1]))
            edge_len=np.sqrt((edge_cr[:,2]-edge_cr[:,0])**2+(edge_cr[:,3]-edge_cr[:,1])**2)
            angle_lsd_thr=pi/36
            reg_lsd=[]
            flag=1
            while len(reg_lsd)<len(edge_cr)-1 and flag<20:
                angle_line_num=np.zeros((18,1),dtype=np.float64)
                for angle_num in range(18):
                    angle1=-pi/2+angle_lsd_thr*(angle_num+1)
                    angle2=angle1+pi/2
                    angle_idx1=np.where(edge_sita<=angle1)
                    angle_idx2=np.where(edge_sita>angle1-angle_lsd_thr)
                    angle_idx3=np.where(edge_sita<=angle2)
                    angle_idx4=np.where(edge_sita>angle2-angle_lsd_thr)
                    angle_idx=set(angle_idx1[0])&set(angle_idx2[0])|set(angle_idx3[0])&set(angle_idx4[0])
                    temp_len=[]
                    temp_len.extend([edge_len[k] for k in angle_idx])
                    temp_len=np.array(temp_len)
                    if len(temp_len)>0:
                        angle_line_num[angle_num]=sum(temp_len)
                max_angle_idx=np.argmax(angle_line_num)
                angle1=-pi/2+angle_lsd_thr*(max_angle_idx+1)
                angle2=angle1+pi/2
                angle_idxt1=set(np.where(edge_sita<angle1+angle_lsd_thr/2)[0])\
                    &set(np.where(edge_sita>angle1-angle_lsd_thr*3/2)[0])
                angle_idxt2=set(np.where(edge_sita<angle2+angle_lsd_thr/2)[0])\
                    &set(np.where(edge_sita>angle2-angle_lsd_thr*3/2)[0])
                temp_line_P=np.array([[edge_sita[k], edge_len[k]] for k in list(angle_idxt1)])
                temp_line_V=np.array([[edge_sita[k], edge_len[k]] for k in list(angle_idxt2)])
                if flag>20:
                    break
                if len(temp_line_P)==0 or len(temp_line_V)==0:
                    flag+=1
                    continue
                temp_sita=(sum(temp_line_P[:,0]*(temp_line_P[:,1]**2))+sum((temp_line_V[:,0]-pi/2)*(temp_line_V[:,1]**2)))\
                    /(sum(temp_line_P[:,1]**2)+sum(temp_line_V[:,1]**2))
                angle_idxt10=[]
                angle_idxt20=[]
                if temp_sita-angle_lsd_thr<-pi/2:
                    angle_idxt10=np.where(edge_sita>-temp_sita-angle_lsd_thr)[0]
                elif temp_sita+angle_lsd_thr>0:
                    angle_idxt20=np.where(edge_sita<temp_sita+angle_lsd_thr-pi/2)[0]
                angle_idxt1=set(np.where(edge_sita<temp_sita+angle_lsd_thr)[0])\
                    &set(np.where(edge_sita>temp_sita-angle_lsd_thr)[0])
                angle_idxt2=set(np.where(edge_sita<temp_sita+angle_lsd_thr+pi/2)[0])\
                    &set(np.where(edge_sita>temp_sita-angle_lsd_thr+pi/2)[0])
                angle_idxt1=angle_idxt1.union(set(angle_idxt10))
                angle_idxt2=angle_idxt2.union(set(angle_idxt20))
                temp_line_P=np.array([[edge_sita[k], edge_len[k]] for k in list(angle_idxt1)])
                temp_line_V=np.array([[edge_sita[k], edge_len[k]] for k in list(angle_idxt2)])
                line_h1=temp_line_P
                line_h2=temp_line_V
                for k, lineh in enumerate(line_h1):
                    lineh[0]=temp_sita
                    reg_lsd.append(np.array([lineh[0],lineh[1],flag]))
                for k, lineh in enumerate(line_h2):
                    lineh[0]=temp_sita+pi/2
                    reg_lsd.append(np.array([lineh[0],lineh[1],flag]))
                for k in sorted(list(angle_idxt1|angle_idxt2))[::-1]:
                    temp_s=list(edge_sita)
                    temp_s.pop(k)
                    edge_sita=np.array(temp_s)
                    temp_l=list(edge_len)
                    temp_l.pop(k)
                    edge_len=np.array(temp_l)
                flag+=1

            reg_lsd_arr=np.array(reg_lsd)
            if len(reg_lsd_arr)>0:

                edge_ori=np.array(list(set(reg_lsd_arr[:,2])))
                edge_oril=edge_ori[:]
                len_thr=200
                for j, ori in enumerate(edge_ori):
                    edge_orim=np.where(reg_lsd_arr[:,2]==ori)[0]
                    edge_oril[j]=sum([reg_lsd_arr[k,1] for k in edge_orim])
                    edge_orix=np.array([edge_ori[k] for k in np.where(edge_oril<len_thr)]).T
                    line_pt=copy.deepcopy(reg_lsd)
                for j in range(len(edge_orix))[::-1]:
                    if line_pt[j][2]==edge_orix[j]:
                        line_pt.pop(j)
                edge_oril0=np.array(list(set(np.array(line_pt)[:,0])))

                angle_lsd_edit=pi/9
                reg_line_lsd=copy.deepcopy(reg_line_s4t)
                ori_edit=np.zeros((len(reg_line_lsd),2),dtype=np.uint8)
                for j, reg_line0 in enumerate(reg_line_lsd):
                    for k, oried in enumerate(edge_oril0):
                        if abs(reg_line0[6]-oried)<angle_lsd_edit and ori_edit[j,1]<1:
                            ori_edit[j,:]=np.array([oried,1])
                            reg_line0[6]=oried
                        elif abs(reg_line0[6]-oried-pi/2)<angle_lsd_edit and ori_edit[j,1]<1:
                            ori_edit[j,:]=np.array([oried+pi/2,1])
                            reg_line0[6]=oried+pi/2
                            if oried+pi/2>pi/2:
                                ori_edit[j,:]=np.array([oried-pi/2,1])
                                reg_line0[6]=oried-pi/2
                        elif abs(reg_line0[6]-oried+pi/2)<angle_lsd_edit and ori_edit[j,1]<1:
                            ori_edit[j,:]=np.array([oried-pi/2,1])
                            reg_line0[6]=oried-pi/2
                            if oried-pi/2<-pi/2:
                                ori_edit[j,:]=np.array([oried+pi/2,1])
                                reg_line0[6]=oried+pi/2
            else:
                reg_line_lsd=copy.deepcopy(reg_line_s4t)

            for k in range(len(reg_line_lsd)):
                for l in range(7):
                    if type(reg_line_lsd[k][l])==np.array:
                        reg_line_lsd[k][l]=float(reg_line_lsd[k][l])
        else:
            reg_line_lsd=copy.deepcopy(reg_line_s4)
        
        
        
        
        #6.line connection
        reg_line_s5=[]
        flag=0
        flag0=0
        if len(reg_line_lsd)<3:
            reg_line_s5=copy.deepcopy(reg_line_s2)
            reg_line_lsd=copy.deepcopy(reg_line_s2)
            reg_line_s5.append(reg_line_lsd[0])
        else:
            reg_line_s5.append(reg_line_lsd[0])
        line_num=len(reg_line_lsd)
        reg_line_lsdm=np.mat(np.reshape(np.array(copy.deepcopy(reg_line_lsd)),(line_num,7)))
        for j in range(1,len(reg_line_lsd)):
            pta=np.mat([reg_line_lsdm[flag,0],reg_line_lsdm[flag,2]]).T
            ptb=np.mat([reg_line_lsdm[flag,1],reg_line_lsdm[flag,3]]).T
            ptc=np.mat([reg_line_lsdm[j,0],reg_line_lsdm[j,2]]).T
            ptd=np.mat([reg_line_lsdm[j,1],reg_line_lsdm[j,3]]).T
            if abs(reg_line_lsd[flag][5]-reg_line_lsd[j][5])>angle_thr/6:
                ts=np.linalg.lstsq((np.hstack(((pta-ptb),(ptd-ptc)))), (pta-ptc))
                ptq=pta+(ptb-pta)*ts[0][0]
                reg_line_s5[flag+flag0][1]=ptq[0]
                reg_line_s5[flag+flag0][3]=ptq[1]
                reg_line_s5.append(reg_line_lsd[j])
                reg_line_s5[flag+flag0+1][0]=ptq[0]
                reg_line_s5[flag+flag0+1][2]=ptq[1]
            else:
                reg_line_s5.append(reg_line_lsd[j])
                reg_line_s5.append(reg_line_lsd[j])
                tx=np.array([float(reg_line_s5[flag+flag0][1]),float(reg_line_lsd[j][0])])
                ty=np.array([float(reg_line_s5[flag+flag0][3]),float(reg_line_lsd[j][2])])
                cori=float(reg_line_lsd[flag][5]+reg_line_lsd[j][5])/2
                if cori>0:
                    cori+=-pi/2
                else:
                    cori+=pi/2
                pm1=np.mat([np.mean(tx)-1,np.mean(ty)-cori]).T
                pm2=np.mat([np.mean(tx)+1,np.mean(ty)+cori]).T
                ts1=np.linalg.lstsq((np.hstack(((pta-ptb),(pm2-pm1)))), (pta-pm1))
                ptq1=pta+(ptb-pta)*ts1[0][0]
                ts2=np.linalg.lstsq((np.hstack(((pm1-pm2),(ptd-ptc)))), (pm1-ptc))
                ptq2=pm1+(pm2-pm1)*ts2[0][0]
                reg_line_s5[flag+flag0][1]=ptq1[0]
                reg_line_s5[flag+flag0][3]=ptq1[1]
                reg_line_s5[flag+flag0+1]=[ptq1[0],ptq2[0],ptq1[1],ptq2[1],reg_line_s5[flag+flag0][4]+flag0+1,\
                                        cori,sqrt((ptq1[0]-ptq2[0])**2+(ptq1[1]+ptq2[1])**2)]
                reg_line_s5[flag+flag0+2][0]=ptq2[0]
                reg_line_s5[flag+flag0+2][2]=ptq2[1]
                line_num+=1
                flag0+=1
            flag+=1 
        
        if line_num>0:
            pta=np.mat([reg_line_lsdm[j,0],reg_line_lsdm[j,2]]).T
            ptb=np.mat([reg_line_lsdm[j,1],reg_line_lsdm[j,3]]).T
            ptc=np.mat([reg_line_lsdm[0,0],reg_line_lsdm[0,2]]).T
            ptd=np.mat([reg_line_lsdm[0,1],reg_line_lsdm[0,3]]).T
            if abs(reg_line_lsd[j][5]-reg_line_lsd[0][5])>angle_thr/6:
                ts=np.linalg.lstsq((np.hstack(((pta-ptb),(ptd-ptc)))), (pta-ptc))
                ptq=pta+(ptb-pta)*ts[0][0]
                reg_line_s5[line_num-1][1]=ptq[0]
                reg_line_s5[line_num-1][3]=ptq[1]
                reg_line_s5[0][0]=ptq[0]
                reg_line_s5[0][2]=ptq[1]
            else:
                reg_line_s5.append(reg_line_lsd[j])
                tx=np.array([float(reg_line_s5[line_num-1][1]),float(reg_line_lsd[0][0])])
                ty=np.array([float(reg_line_s5[line_num-1][3]),float(reg_line_lsd[0][2])])
                cori=float(reg_line_lsd[0][5]+reg_line_lsd[j][5])/2
                if cori>0:
                    cori+=-pi/2
                else:
                    cori+=pi/2
                pm1=np.mat([np.mean(tx)-1,np.mean(ty)-cori]).T
                pm2=np.mat([np.mean(tx)+1,np.mean(ty)+cori]).T
                ts1=np.linalg.lstsq((np.hstack(((pta-ptb),(pm2-pm1)))), (pta-pm1))
                ptq1=pta+(ptb-pta)*ts1[0][0]
                ts2=np.linalg.lstsq((np.hstack(((pm1-pm2),(ptd-ptc)))), (pm1-ptc))
                ptq2=pm1+(pm2-pm1)*ts2[0][0]
                reg_line_s5[flag+flag0][1]=ptq1[0]
                reg_line_s5[flag+flag0][3]=ptq1[1]
                reg_line_s5[flag+flag0+1]=[ptq1[0],ptq2[0],ptq1[1],ptq2[1],reg_line_s5[flag+flag0][4]+flag0+1,\
                                        cori,sqrt((ptq1[0]-ptq2[0])**2+(ptq1[1]+ptq2[1])**2)]
                reg_line_s5[0][0]=ptq2[0]
                reg_line_s5[0][2]=ptq2[1]
                line_num+=1
                flag0+=1

        reg_line_final=copy.deepcopy(reg_line_s5)
        reg_line_final.append(copy.deepcopy(reg_line_final[0]))
        max_angle_idx=np.argmax(np.reshape(np.array(copy.deepcopy(reg_line_final)),(len(reg_line_final),7))[:,6])
        ori_line=reg_line_final[max_angle_idx][5]
        if ori_line>pi/4:
            ori_line-=pi/2
        elif ori_line<-pi/4:
            ori_line+=pi/2
        main_ori[i]=ori_line
        
        pt_temp=[]
        for j in range(len(reg_line_final)-1):
            reg_line_final[j][4]=j
            pt_temp.append([i,float(reg_line_final[j][1]+reg_line_final[j+1][0])/2,\
                            float(reg_line_final[j][3]+reg_line_final[j+1][2])/2])
        pt0.append(pt_temp)

        for k in range(len(reg_line_final)):
            for l in range(7):
                if type(reg_line_final[k][l])==np.matrix:
                    reg_line_final[k][l]=float(reg_line_final[k][l])
        reg_line_final = [list(t) for t in set(tuple(_) for _ in reg_line_final)]
        reg_line_final.append(copy.deepcopy(reg_line_final[0]))

        for j, reg_linef in enumerate(reg_line_final):
            reg_linef.insert(0,i)

        reg_line.append(reg_line_final)
    return reg_line,pt0,main_ori
        
def merge_line_function(reg_line_p,reg_line_n,pointer_p,pointer_n):
    tx=[reg_line_n[pointer_n][0],reg_line_p[pointer_p][1]]
    ty=[reg_line_n[pointer_n][2],reg_line_p[pointer_p][3]]
    if pointer_p == len(reg_line_p) - 1 and pointer_n == 0:
        tx = [reg_line_n[pointer_n][1], reg_line_p[pointer_p][0]]
        ty = [reg_line_n[pointer_n][3], reg_line_p[pointer_p][2]]
    if tx[1]==tx[0]:
        if ty[1]>ty[0]:
            ori=pi/2
        else:
            ori=-pi/2
    else:
        ori=np.arctan((ty[1]-ty[0])/(tx[1]-tx[0]))
    reg_line_re=[tx[0],tx[1],ty[0],ty[1],float(reg_line_n[pointer_n][4]),ori,sqrt((tx[0]-tx[1])**2+(ty[0]-ty[1])**2)]
    return reg_line_re

def merge_line_function_dir(temp_line,reg_line_n,pointer_n):
    dx=reg_line_n[pointer_n][1]-temp_line[0]
    dy=reg_line_n[pointer_n][3]-temp_line[2]
    if reg_line_n[pointer_n][6]+temp_line[6]!=0:
        newx1=reg_line_n[pointer_n][0]-temp_line[6]/(reg_line_n[pointer_n][6]+temp_line[6])*dx
        newy1=reg_line_n[pointer_n][2]-temp_line[6]/(reg_line_n[pointer_n][6]+temp_line[6])*dy
        newx2=temp_line[1]+reg_line_n[pointer_n][6]/(reg_line_n[pointer_n][6]+temp_line[6])*dx
        newy2=temp_line[3]+reg_line_n[pointer_n][6]/(reg_line_n[pointer_n][6]+temp_line[6])*dy
    else:
         newx1=reg_line_n[pointer_n][0]
         newy1=reg_line_n[pointer_n][2]
         newx2=temp_line[1]
         newy2=temp_line[3]
    reg_line_re=[newx1,newx2,newy1,newy2,reg_line_n[pointer_n][4],\
                 reg_line_n[pointer_n][5],reg_line_n[pointer_n][6]+temp_line[6]]
    return reg_line_re


