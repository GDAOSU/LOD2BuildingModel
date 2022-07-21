import math

from skimage import io,data,filters,segmentation,measure,morphology,color,feature,draw
from skimage.util import img_as_ubyte
from scipy import ndimage
import numpy as np
from math import pi,sqrt,sin,cos,log,asin
import cv2
import copy
import rdp
from bresenham import bresenham
from scipy import optimize
import cmath
import multiprocessing
from functools import partial
import functools
import random
import os

from building_polygon import *
from building_decomposition import *
from building_refinement import *

import warnings
warnings.filterwarnings("ignore")

def calculate_circulometry(label_mask):
    #circulometry
    ret, thresh = cv2.threshold(label_mask*200, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    circulometry = (perimeter**2)/(4*pi*area)

    return circulometry

def calc_R(xc, yc, xi, yi):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((xi-xc)**2 + (yi-yc)**2)

def CircleLeastSquare(c,xi,yi):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c,xi,yi)
    return Ri - Ri.mean()

def one_circle_hough_trans(label_mask,label_num,Thr_line):
    #ini_line = initial_line_extraction(label_mask,label_num)
    imgsize = np.shape(label_mask)
    dp_line = []
    grid_para = 4
    grident_para = 0.2
    circle_detect_para = 5
    pad = 10
    circle_radian_list = []
    for lnum in range(1, label_num + 1):
    #for lnum in range(1, 60):
    #for lnum in range(214, 250):#range(240, 250):#range(521, 544):
        img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        loc_mask_xy = np.where(label_mask == lnum)
        min_locy = np.min(loc_mask_xy[0])
        max_locy = np.max(loc_mask_xy[0])
        min_locx = np.min(loc_mask_xy[1])
        max_locx = np.max(loc_mask_xy[1])
        img_mask[loc_mask_xy] = 1
        img_mask_loc = img_mask[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        # 1. initial boundary/keypoints detection
        # Douglas-Peucker algorithm
        contours = measure.find_contours(img_mask_loc, 0.5)
        dp_pt = rdp(contours[0], epsilon=1, algo="iter", return_mask=False)
        temp_line = []
        pt_key = []
        pt_sdis = []
        for j in range(len(dp_pt) - 1):
            pt1 = np.array(dp_pt[j])
            pt2 = np.array(dp_pt[j + 1])
            pt1[0], pt1[1] = pt1[1], pt1[0]
            pt2[0], pt2[1] = pt2[1], pt2[0]
            angle_line = np.arctan((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
            temp_line.append([angle_line, pt1, pt2, j])
            pt_key.append(pt1)
            pt_sdis.append(sqrt((pt2[1] - pt1[1])**2+(pt2[0] - pt1[0])**2))
        dp_line.append(temp_line)
        if len(temp_line) < circle_detect_para * 2:
            continue

        # 2. circle center computation
        #calculate median of lines
        med = []
        for j, tline in enumerate(temp_line):
            med_xy = np.array([(tline[1][0]+tline[2][0])/2,(tline[1][1]+tline[2][1])/2])
            med_angle = -1/tline[0]
            med.append(np.array([med_angle,med_xy[0],med_xy[1],j]))
        #create coarse grid, reduce the detection error
        imgsize_loc = img_mask_loc.shape
        img_grid = np.zeros((ceil(imgsize_loc[0]/grid_para),ceil(imgsize_loc[1]/grid_para)),dtype=np.uint8)
        #compute the pixels covered by vertical line
        for j, tmed in enumerate(med):
            med_g = tmed[1:3]/4
            #compute the coordinate of image boundary
            pt_boundary = np.zeros((4,2),dtype=float)
            pt_boundary[0, 0], pt_boundary[1, 0] = 0, img_grid.shape[1] - 1
            pt_boundary[0, 1], pt_boundary[1, 1] = med_g[1] - med[j][0]*med_g[0], med[j][0] * (img_grid.shape[1] - 1) + med_g[1] - med[j][0] * med_g[0]
            pt_boundary[2, 0], pt_boundary[3, 0] = med_g[0] - med_g[1]/med[j][0], med_g[0] + (img_grid.shape[0] - 1 - med_g[1])/med[j][0]
            pt_boundary[2, 1], pt_boundary[3, 1] = 0, img_grid.shape[0] - 1
            #determine intersection
            pt_sort = pt_boundary[np.argsort(pt_boundary[:,0])]
            pt_b1 = pt_sort[1,:]
            pt_b2 = pt_sort[2,:]

            line_pixel = list(bresenham(round(pt_b1[0]),round(pt_b1[1]),round(pt_b2[0]),round(pt_b2[1])))
            line_pixel_ex = list(np.where(np.array(line_pixel)[:, 1] < 0)[0])
            line_pixel_ex.extend(list(np.where(np.array(line_pixel)[:, 1] > img_grid.shape[0] - 1)[0]))
            line_pixel_ex.extend(list(np.where(np.array(line_pixel)[:, 0] > img_grid.shape[1] - 1)[0]))
            line_pixel_ex.extend(list(np.where(np.array(line_pixel)[:, 0] < 0)[0]))
            line_pixel_ex.sort(reverse = True)
            for index_del in line_pixel_ex:
                line_pixel.pop(index_del)
            line_pixel = np.array(line_pixel)[:, [1, 0]]
            img_grid[tuple(line_pixel.T)]+=1

        center_coarse_grid = np.where(img_grid>=Thr_line)

        if len(center_coarse_grid[0]) < 1:
            continue
        elif len(center_coarse_grid[0]) > 20:
            center_coarse_grid = np.where(img_grid >= Thr_line + 1)

        # 3. radius computation & uncandidate circle exclusion
        circle_coarse_para = []
        circle_coarse_pt = []
        print(lnum)
        center_coarse_grid = tuple(np.array(center_coarse_grid).T)
        pt_key_ext = copy.deepcopy(pt_key)
        pt_key_ext.extend(pt_key_ext[0:circle_detect_para - 1])
        num_order = list(range(len(pt_key)))
        num_order.extend(range(circle_detect_para - 1))
        pt_key_ext = np.vstack((np.array(num_order), np.array(pt_key_ext).T)).T

        for center_coarseg in center_coarse_grid:
            center_coarse = center_coarseg * grid_para + grid_para/2
            center_coarse[0], center_coarse[1] = center_coarse[1], center_coarse[0]
            pt_distance = []
            pt_dist_diff = []
            pt_dist_gradient = []
            # use gradient of distance to circle center
            for j, tline in enumerate(temp_line):
                pt_distance.append(sqrt((center_coarse[0] - pt_key[j][0])**2\
                                        +(center_coarse[1] - pt_key[j][1])**2))
            for j, dist in enumerate(pt_distance):
                nxt = j + 1
                if j == len(pt_distance) -1:
                    nxt = 0
                temp_diff = dist - pt_distance[nxt]
                pt_dist_diff.append(temp_diff)
                pt_dist_gradient.append(temp_diff/pt_sdis[j])
            pt_dist_diff.extend(pt_dist_diff[0:circle_detect_para - 1])
            pt_dist_gradient.extend(pt_dist_gradient[0:circle_detect_para - 1])

            coarse_radius_list = []
            coarse_pt_list = []
            for j in range(len(pt_distance)):
                temp_grad = np.array(pt_dist_gradient[j:j+circle_detect_para])
                avg_grad = np.mean(abs(temp_grad))
                out_grad = len(np.where(abs(temp_grad)>grident_para))
                if avg_grad < grident_para/2 and out_grad < 2:
                    coarse_radius = np.mean(np.array(pt_distance[j:j+circle_detect_para]))
                    coarse_radius_list.append(coarse_radius)
                    coarse_pt_list.append(pt_key_ext[j:j+circle_detect_para])
            if coarse_radius_list:
                circle_coarse_para.append([center_coarse[0], center_coarse[1], np.array(coarse_radius_list)])
                circle_coarse_pt.append(coarse_pt_list)

        if not circle_coarse_para:
            continue

        circle_coarse_para_ex = []
        for j, circle in enumerate(circle_coarse_para):
            center = np.array(circle[0:2])
            circle_info = sorted(zip(circle[2],circle_coarse_pt[j]))
            radius, pt_list = zip(*circle_info)
            radius_tmp = radius[0]
            circle_radius = []
            circle_pt = []
            pt_tmp = []
            flagsum = 0
            for k in range(len(radius)):
                if radius[k] < radius_tmp + pad:
                    radius_tmp = (radius[k]+radius_tmp*flagsum)/(flagsum+1)
                    pt_tmp.extend(pt_list[k])
                    flagsum = flagsum + 1
                else:
                    circle_radius.append(radius_tmp)
                    pt_tmpt = np.array(list(set([tuple(t) for t in pt_tmp])))
                    circle_pt.append(pt_tmpt[np.argsort(pt_tmpt[:,0])])
                    radius_tmp = copy.deepcopy(radius[k])
                    pt_tmp = []
                    pt_tmp.extend(pt_list[k])
                    flagsum = 1
            circle_radius.append(radius_tmp)
            pt_tmpt = np.array(list(set([tuple(t) for t in pt_tmp])))
            circle_pt.append(pt_tmpt[np.argsort(pt_tmpt[:, 0])])
            circle_coarse_para_ex.append([center,circle_radius,circle_pt])

        #self check of coarse circle
        circle_coarse_para_ex0 = []
        coasre_pt_num = []
        for j, circle in enumerate(circle_coarse_para_ex):
            center = circle[0]
            pt_num_temp = 0
            for k, rad_coarse in enumerate(circle[1]):
                pt_num_temp += len(circle[2][k])
            remain_flag = 1
            if circle_coarse_para_ex0:
                for k, circle_temp in enumerate(circle_coarse_para_ex0):
                    dist_temp = np.sqrt((center[0] - circle_temp[0][0])**2 + (center[1] - circle_temp[0][1])**2)
                    if dist_temp < 20:
                        if pt_num_temp > coasre_pt_num[k]:
                            circle_coarse_para_ex0.pop(k)
                            coasre_pt_num.pop(k)
                        else:
                            remain_flag = 0
            if remain_flag:
                circle_coarse_para_ex0.append(circle)
                coasre_pt_num.append(pt_num_temp)
        circle_coarse_para_ex = circle_coarse_para_ex0


        # 4. fine parameter computation
        circle_fine_para = []
        for j, circle in enumerate(circle_coarse_para_ex):
            center = circle[0]
            center_concentric_list = []
            for k, rad_coarse in enumerate(circle[1]):
                pt_c = circle[2][k][:,1:3]
                center_concentric, ier = optimize.leastsq(CircleLeastSquare, tuple(center), args=(pt_c[:,0],pt_c[:,1]))
                center_concentric_list.append(center_concentric)
            center_fine = np.array(center_concentric_list).mean(axis=0)
            radius_concentric_list = []
            for k, rad_coarse in enumerate(circle[1]):
                pt_c = circle[2][k][:,1:3]
                radius_concentric = calc_R(center_fine[0], center_fine[1], pt_c[:,0], pt_c[:,1]).mean()
                radius_concentric_list.append(radius_concentric)
            circle_fine_para.append([center_fine,radius_concentric_list])

        # 5. radian compuation
        circle_radian_para = []
        xy_loc2org = np.array([min_locx - pad, min_locy - pad])
        pt_cdis = list(copy.deepcopy(pt_sdis))
        pt_cdis.extend(pt_cdis[0:circle_detect_para])
        pt_cdis = np.array(pt_cdis)
        for j, circle in enumerate(circle_fine_para):
            center = circle[0]
            radian_list = []
            for k, radius in enumerate(circle[1]):
                pt_c = pt_key_ext[:,1:3]
                pt_candidate = []
                for temp in range(len(pt_key)):
                    pt_temp = pt_c[temp:temp+circle_detect_para, :]
                    pt_cdis_temp = pt_cdis[temp:temp+circle_detect_para-1]
                    radius_dist = calc_R(center[0], center[1], pt_temp[:, 0], pt_temp[:, 1]) - radius
                    radius_grad = (radius_dist[0:-1] - radius_dist[1:])/pt_cdis_temp
                    qual_num = np.where(abs(radius_dist) < 5)[0]
                    if np.max(abs(radius_dist)) > 8 or np.max(abs(radius_grad)) > grident_para * 1.5:
                        continue
                    if abs(radius_dist).mean() < 3 and len(qual_num) >= Thr_line - 1:
                        pt_candidate.extend(pt_key_ext[temp:temp+circle_detect_para, :])
                pt_candidate = np.array(list(set([tuple(t) for t in pt_candidate])))
                if len(pt_candidate) < 1:
                    continue
                pt_candidate = pt_candidate[np.argsort(pt_candidate[:, 0])]
                radius_dist_check = calc_R(center[0], center[1], pt_candidate[:, 1], pt_candidate[:, 2]) - radius
                radian = 2 * pi
                pt_first = pt_candidate[0, 1:3]
                pt_last = pt_candidate[0, 1:3]
                if len(pt_candidate) < min(0.9 * len(pt_key),len(pt_key)-1):
                    if abs(pt_candidate[0, 0] - pt_candidate[-1, 0])+1 == len(pt_key):
                        lack_num = list(set(range(len(pt_key))) - set(list(pt_candidate[:,0])))
                        pt_candidate[:lack_num[0], 0] += len(pt_key)
                        pt_candidate = np.array(list(set([tuple(t) for t in pt_candidate])))
                    arc_first = pt_candidate[0,1:3]
                    arc_last = pt_candidate[-1,1:3]
                    #translate to polar coordinate to avoid distance error
                    polar_first = cmath.polar(complex(arc_first[0] - center[0],arc_first[1] - center[1]))
                    polar_last = cmath.polar(complex(arc_last[0] - center[0], arc_last[1] - center[1]))
                    radian = polar_last[1] - polar_first[1]
                    pt_first = cmath.rect(radius, polar_first[1])
                    pt_first = np.array([pt_first.real + center[0], pt_first.imag + center[1]])
                    pt_last = cmath.rect(radius, polar_last[1])
                    pt_last = np.array([pt_last.real + center[0], pt_last.imag + center[1]])
                    if len(pt_candidate) > len(pt_distance) and radian < pi:
                        radian = pi*2 - radian
                    if abs(radian) < pi/2:
                        continue
                    elif abs(radian) > 3*pi/2:
                        radian = 2 * pi
                        pt_first = pt_candidate[0, 1:3]
                        pt_last = pt_candidate[0, 1:3]
                radian_list.append([radian, pt_first + xy_loc2org, pt_last + xy_loc2org])

            if len(radian_list) < 1:
                continue
            circle_radian_para.append([lnum,center + xy_loc2org, circle[1], radian_list])
        if len(circle_radian_para) < 1:
            continue
        circle_radian_list.append(circle_radian_para)

    return circle_radian_list

#small circle check in 2D
def circle_check(circle_radian, rectangle, label_mask, orthot, dsmt):
    imgsize = np.shape(label_mask)
    pad = 10
    Tl = 90
    Td = 10
    Th1 = 0.5
    Th2 = 0.1
    Th_overlap1 = 0.5
    Th_overlap2 = 0.2
    circle_radian_out = []
    rectangle_ex = []
    rectangle_out = copy.deepcopy(rectangle)
    #check small circle & rectangle
    for cir_num, circle_l in enumerate(circle_radian):
        lnum = circle_l[0][0]
        img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        loc_mask_xy = np.where(label_mask == lnum)
        min_locy = np.min(loc_mask_xy[0])
        max_locy = np.max(loc_mask_xy[0])
        min_locx = np.min(loc_mask_xy[1])
        max_locx = np.max(loc_mask_xy[1])
        img_mask[loc_mask_xy] = 1
        img_mask_loc = img_mask[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        xy_loc2org = np.array([min_locx - pad, min_locy - pad])
        rect = rectangle[lnum - 1]
        if len(rect) == 0 or len(circle_l) != 1 or len(rect) > 2:
            circle_radian_out.append(circle_l)
            continue
        (m_loc, n_loc) = np.shape(img_mask_loc)
        rect_c = rect[0]
        rectx, recty = rect_c[2:9:2] - xy_loc2org[0], rect_c[3:10:2] - xy_loc2org[1]
        rectx[3], rectx[2] = rectx[2], rectx[3]
        recty[3], recty[2] = recty[2], recty[3]
        rec_mask = draw.polygon2mask([m_loc, n_loc], np.vstack((recty, rectx)).T).astype(float)
        if len(rect) == 2:
            rect_c = rect[1]
            rectx, recty = rect_c[2:9:2] - xy_loc2org[0], rect_c[3:10:2] - xy_loc2org[1]
            rectx[3], rectx[2] = rectx[2], rectx[3]
            recty[3], recty[2] = recty[2], recty[3]
            rec_mask1 = draw.polygon2mask([m_loc, n_loc], np.vstack((recty, rectx)).T).astype(float)
            rec_mask = rec_mask + rec_mask1
            rec_mask[np.where(rec_mask > 1)] = 1
        intersection_rec = np.logical_and(img_mask_loc, rec_mask)
        union_rec = np.logical_or(img_mask_loc, rec_mask)
        iou_rec = np.sum(intersection_rec) / np.sum(union_rec)
        circle = circle_l[0]
        center = circle[1]
        if len(circle[3]) > 1:
            circle_radian_out.append(circle_l)
            continue
        radius = circle[2][0]
        #mask awareness
        circle_mask = img_mask_loc * 0
        center_loc = center - xy_loc2org
        circle_mask = cv2.circle(circle_mask, (round(center_loc[0]), round(center_loc[1])), round(radius), 1.0, -1)
        intersection_cir = np.logical_and(img_mask_loc, circle_mask)
        union_cir = np.logical_or(img_mask_loc, circle_mask)
        iou_cir = np.sum(intersection_cir) / np.sum(union_cir)
        circulometry = calculate_circulometry(img_mask_loc)
        if iou_cir < iou_rec * 1.8 and circulometry > 1.1 and np.sum(img_mask_loc) < 1000:
            continue
        circle_radian_out.append(circle_l)
        rectangle_ex.append(lnum - 1)
    for i_ex in rectangle_ex:
        rectangle_out[i_ex] = []

    #check overlap between circle & rectangle
    rectangle_out1 = copy.deepcopy(rectangle_out)
    for cir_num, circle_l in enumerate(circle_radian_out):
        lnum = circle_l[0][0]
        img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        loc_mask_xy = np.where(label_mask == lnum)
        min_locy = np.min(loc_mask_xy[0])
        max_locy = np.max(loc_mask_xy[0])
        min_locx = np.min(loc_mask_xy[1])
        max_locx = np.max(loc_mask_xy[1])
        img_mask[loc_mask_xy] = 1
        img_mask_loc = img_mask[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        xy_loc2org = np.array([min_locx - pad, min_locy - pad])
        rect = rectangle_out[lnum - 1]
        if len(rect) < 1:
            continue
        rectangle_temp = []
        center = circle[1]
        (m_loc, n_loc) = np.shape(img_mask_loc)
        re_detect = 0
        circle_mask0 = img_mask_loc * 0
        for connum, concircle in enumerate(circle[3]):
            radius = circle[2][connum]
            radian_first, radian_last, radian = concircle[1] - xy_loc2org, concircle[2] - xy_loc2org, concircle[0]
            #mask awareness
            circle_mask = img_mask_loc * 0
            center_loc = center - xy_loc2org
            circle_mask = cv2.circle(circle_mask, (round(center_loc[0]), round(center_loc[1])), round(radius), 1.0, -1)

            # notice the axis direction of polar coordinates
            polar_first = cmath.polar(complex(radian_first[0] - center_loc[0], radian_first[1] - center_loc[1]))
            polar_last = cmath.polar(complex(radian_last[0] - center_loc[0], radian_last[1] - center_loc[1]))
            ori_first, ori_last = polar_first[1], polar_last[1]
            if ori_first < 0:
                ori_first += pi * 2
            if ori_last < ori_first:
                ori_last += pi * 2

            # if ori_last == ori_first, this is a full circle, otherwise, it is a circle with radian
            if ori_last != ori_first:
                circle_mask_cover = radian_mask(center_loc, radius, ori_first, ori_last, circle_mask)
            else:
                circle_mask_cover = circle_mask
            circle_mask0[np.where(circle_mask_cover > 0)] = 1
            for recnum, rect_c in enumerate(rect):
                rectx, recty = rect_c[2:9:2] - xy_loc2org[0], rect_c[3:10:2] - xy_loc2org[1]
                rec_mask = draw.polygon2mask([m_loc, n_loc], np.vstack((recty, rectx)).T).astype(float)
                inters_mask = circle_mask_cover * img_mask_loc * rec_mask
                rec_int_mask = img_mask_loc * rec_mask
                if np.sum(inters_mask) < (1 - Th_overlap1) * np.sum(rec_int_mask):
                    rectangle_temp.append(rect_c)
                    continue
                if np.sum(inters_mask) > Th_overlap2 * np.sum(rec_int_mask):
                    re_detect = 1
        if re_detect == 1:
            new_mask = img_mask_loc - circle_mask0
            new_mask[np.where(new_mask < 0)] = 0
            new_label_mask = label_mask * 0
            new_label_mask[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad] = new_mask
            #re-detect line
            ini_linec = initial_line_extraction(new_label_mask, 1)
            adj_linec = line_adjustment(ini_linec, 1)
            line_reg_thr = Tl
            reg_linec, pt0, main_ori = line_regularization(adj_linec, 1, line_reg_thr, orthot)
            #re-detect line
            pt_list_int = []
            pt_list_float = []
            line_list_int = []
            for ic in range(1):
                pt_int = copy.deepcopy(pt0[ic])
                pt_int.append(pt_int[0])
                pt_int = np.array(pt_int).astype(int)
                pt_list_int.append(list(pt_int))
                pt_float = copy.deepcopy(pt0[ic])
                pt_float.append(pt_float[0])
                pt_list_float.append(list(pt_float))
                line_int = []
                for j in range(len(pt0[ic])):
                    k = 1
                    sita = np.arctan((pt_int[j, 2] - pt_int[j + 1, 2]) / (pt_int[j, 1] - pt_int[j + 1, 1]))
                    if abs(sita - main_ori[ic]) < pi / 4:
                        k = 2
                    line_int.append([pt_int[j][0], pt_int[j][1], pt_int[j][2], \
                                     pt_int[j + 1][1], pt_int[j + 1][2], k, sita])
                line_list_int.append(line_int)
            polygon_line = reg_linec
            decp_rectc = decpose_rec(pt_list_int, pt_list_float, line_list_int, polygon_line, dsmt, orthot,
                                    new_label_mask, Td, Th1, Th2)
            rectangle_out1[lnum - 1] = decp_rectc[0]
            for recout in rectangle_out1[lnum - 1]:
                recout[0] += lnum - 1
        elif len(rectangle_temp) > 0 and re_detect == 0:
            rectangle_out1[lnum - 1] = rectangle_temp

    return circle_radian_out, rectangle_out1



#get height via different direction from center
def height_direction_propagation(dir, dsm, center, mask):
    dir_list = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    dirxy = np.array(dir_list[int(dir)])
    center_int = np.array([round(center[0]), round(center[1])])
    height_list = []
    pt_current = center_int.copy()
    while mask[pt_current[1], pt_current[0]] > 0:
        height_list.append(dsm[pt_current[1], pt_current[0]])
        pt_current += dirxy
    return height_list

#RANSAC line fitting & least square line fitting
def fit_ransac_ls_line(pt_list, sigma, iters=1000, P=0.99):
    # sigma: bandwith of RANSAC inlier
    line_a = 0
    line_b = 0
    num_inner = 0

    #RANSAC to fit line & inner points
    for i in range(iters):
        pt_inner_temp = []
        sample_idx = random.sample(range(len(pt_list)), 2)
        x_1 = pt_list[sample_idx[0]][0]
        y_1 = pt_list[sample_idx[0]][1]
        x_2 = pt_list[sample_idx[1]][0]
        y_2 = pt_list[sample_idx[1]][1]
        if x_2 == x_1:
            continue
        a = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - a * x_1
        num_inlier = 0
        for index in range(len(pt_list)):
            y_estimate = a * pt_list[index][0] + b
            if abs(y_estimate - pt_list[index][1]) < sigma:
                num_inlier += 1
                pt_inner_temp.append(pt_list[index])
        if num_inlier > num_inner:
            iters = log(1 - P) / log(1 - pow(num_inlier / len(pt_list), 2))
            num_inner = num_inlier
            line_a = a
            line_b = b
            pt_inner = np.array(pt_inner_temp)
        if num_inlier > len(pt_list) * 0.9:
            break

    #least square to fit accurate line
    pt_inner_x, pt_inner_y = np.array(pt_inner)[:,0], np.array(pt_inner)[:,1]
    line_a = ((pt_inner_x*pt_inner_y).mean() - pt_inner_x.mean()* pt_inner_y.mean())/(pow(pt_inner_x,2).mean()-pow(pt_inner_x.mean(),2))
    line_b = pt_inner_y.mean() - line_a * pt_inner_x.mean()

    return line_a, line_b

#RANSAC circle fitting & least square circle fitting
def fit_ransac_ls_circle(pt_list, sigma, iters=1000, P=0.99):
    # sigma: bandwith of RANSAC inlier
    # circle formula: x^2+(y-b)^2=c^2
    circle_b = 0
    circle_c = 1
    num_inner = 0

    #RANSAC to fit circle & inner points
    for i in range(iters):
        pt_inner_temp = []
        sample_idx = random.sample(range(len(pt_list)), 2)
        x_1 = pt_list[sample_idx[0]][0]
        y_1 = pt_list[sample_idx[0]][1]
        x_2 = pt_list[sample_idx[1]][0]
        y_2 = pt_list[sample_idx[1]][1]
        if x_2 == x_1:
            continue
        b = (x_1 ** 2 + y_1 ** 2 - x_2 ** 2 - y_2 ** 2) / (2 * (y_1 - y_2))
        c = sqrt(x_1 ** 2 + (y_1 - b) ** 2)

        num_inlier = 0
        for index in range(len(pt_list)):
            diff_estimate = c - sqrt(pt_list[index][0] ** 2 + (pt_list[index][1] - b) ** 2)
            if abs(diff_estimate) < sigma:
                num_inlier += 1
                pt_inner_temp.append(pt_list[index])
        if num_inlier > num_inner:
            iters = log(1 - P) / log(1 - pow(num_inlier / len(pt_list), 2))
            num_inner = num_inlier
            circle_b = b
            circle_c = c
            pt_inner = np.array(pt_inner_temp)
        if num_inlier > len(pt_list) * 0.9:
            break

    #least square to fit accurate circle
    pt_inner_x, pt_inner_y = np.array(pt_inner)[:,0], np.array(pt_inner)[:,1]
    pt_center = [0, circle_b]
    center_ls, ier = optimize.leastsq(CircleLeastSquare, tuple(pt_center), args=(pt_inner_x, pt_inner_y))
    radius_ls = calc_R(center_ls[0], center_ls[1], pt_inner_x, pt_inner_y).mean()
    circle_b = center_ls[1]
    circle_c = radius_ls

    return circle_b, circle_c

#3D reconstruction
def circle_reconstruction(dsm, label_mask, circle_radian):
    imgsize = np.shape(label_mask)
    pad = 10
    Th_grad1 = 0.2
    Th_grad2 = 0.2
    circle_roof_para = []
    max_height = np.max(dsm)
    for circle_l in circle_radian:
        lnum = circle_l[0][0]
        img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        loc_mask_xy = np.where(label_mask == lnum)
        min_locy = np.min(loc_mask_xy[0])
        max_locy = np.max(loc_mask_xy[0])
        min_locx = np.min(loc_mask_xy[1])
        max_locx = np.max(loc_mask_xy[1])
        img_mask[loc_mask_xy] = 1
        img_mask_loc = img_mask[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        dsm_loc = dsm[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        dsm_mask = dsm_loc * img_mask_loc
        xy_loc2org = np.array([min_locx - pad, min_locy - pad])
        pro_dir = [0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4]
        roof_para_con = []
        for circle in circle_l:
            center = circle[1]
            for connum, concircle in enumerate(circle[3]):
                radius = circle[2][connum]
                radian_first, radian_last, radian = concircle[1] - xy_loc2org, concircle[2] - xy_loc2org, concircle[0]

                #mask awareness
                circle_mask = img_mask_loc * 0
                center_loc = center - xy_loc2org
                circle_mask = cv2.circle(circle_mask, (round(center_loc[0]), round(center_loc[1])), round(radius), 1.0, -1)
                inters_mask = circle_mask * img_mask_loc
                #notice the axis direction of polar coordinates
                polar_first = cmath.polar(complex(radian_first[0] - center_loc[0], radian_first[1] - center_loc[1]))
                polar_last = cmath.polar(complex(radian_last[0] - center_loc[0], radian_last[1] - center_loc[1]))
                ori_first, ori_last = polar_first[1], polar_last[1]
                if ori_first < 0:
                    ori_first += pi * 2
                if ori_last < 0:
                    ori_last += pi * 2

                #pro_dir = [0, pi/4, pi/2, 3*pi/4, pi, -3*pi/4, -pi/2, -pi/4]
                val_dir = []
                for tdir in pro_dir:
                    if tdir > ori_first and tdir < ori_last:
                        val_dir.append(tdir/(pi/4))
                if ori_first == ori_last:
                    val_dir.extend(list(range(8)))
                val_height = []
                val_grad = []
                val_grad2 = []
                #calculate height and gradient for each direction
                for dir in val_dir:
                    # direction propagation
                    height_list = np.array(height_direction_propagation(dir, dsm_loc, center_loc, inters_mask))
                    if len(height_list) > 10:
                        denoise_num = int(len(height_list)/10)
                        height_list = height_list[0:-denoise_num]

                    grad_list = height_list[0:-1] - height_list[1:]
                    grad2_list = grad_list[0:-1] - grad_list[1:]
                    val_height.append(height_list)
                    val_grad.append(grad_list)
                    val_grad2.append(grad2_list)
                #calculate the mean gradient & height
                ht_grad = np.zeros((len(val_dir), 4), dtype=float)
                diagonal_list = []
                for val in range(len(val_dir)):
                    height_list = val_height[val]
                    grad_list = val_grad[val]
                    grad2_list = val_grad2[val]
                    #normalize the diagonal direction gradient (sqrt(2)->1)
                    diagonal = (val_dir[val] % 2) * (sqrt(2) - 1) + 1
                    ht_grad[val, 0] = np.mean(height_list)
                    ht_grad[val, 1] = np.mean(abs(grad_list)) / diagonal
                    ht_grad[val, 2] = np.mean(abs(grad2_list)) / diagonal
                    ht_grad[val, 3] = np.mean(grad_list) / diagonal
                    diagonal_list.append(diagonal)
                ht_grad_all = np.mean(ht_grad, 0)

                #roof type determination (decision tree)
                max_height = np.max(dsm_mask)
                if ht_grad_all[1] < Th_grad1 or abs(ht_grad_all[3]) < Th_grad1 / 2:
                    roof_type = 0
                else:
                    if ht_grad_all[2] < Th_grad2:
                        roof_type = 1
                    else:
                        roof_type = 2

                # roof parameter calculation
                # parameter structure: [roof type, center height, edge height, radius (for sphere)]
                roof_para = [lnum, connum, 0, ht_grad_all[0], ht_grad_all[0], 0] # parameter for flat
                if roof_type > 0:
                    pt_ht = []
                    if len(val_height) < 1:
                        continue
                    for list_num, htlist in enumerate(val_height):
                        htarray = np.zeros((len(htlist), 2), dtype=float)
                        diagonal = diagonal_list[list_num]
                        htarray[:, 0] = np.array(range(len(htlist))) * diagonal
                        htarray[:, 1] = np.array(htlist)
                        pt_ht.extend(htarray)
                    if roof_type == 1:
                        # use RANSAC to fit the line from height & gradient
                        # ax + b = y
                        line_a, line_b = fit_ransac_ls_line(np.array(pt_ht), Th_grad1)
                        ht_center = line_b
                        ht_edge = line_a * radius + line_b
                        roof_para = [lnum, connum, roof_type, ht_center, ht_edge, 0]
                    elif roof_type == 2:
                        # use RANSAC to fit the circle from gradient
                        # x^2 + (y-b)^2 = c^2
                        circle_b, circle_c = fit_ransac_ls_circle(np.array(pt_ht), Th_grad1)
                        ht_center = circle_b + circle_c
                        ht_edge = sqrt(circle_c ** 2 - radius ** 2) + circle_b
                        if ht_center > max_height:
                            continue
                        roof_para = [lnum, connum, roof_type, ht_center, ht_edge, circle_c]
                roof_para_con.append(roof_para)
            circle_roof_para.append(roof_para_con)

    return circle_roof_para

def radian_mask(center, radius, ori_first, ori_last, mask):
    (m, n) = np.shape(mask)
    mask_out = np.zeros((m, n), dtype=np.uint8)
    for row in range(m):
        for col in range(n):
            dis = sqrt((row - center[1])**2+(col - center[0])**2)
            if dis > radius:
                continue
            polar_temp = cmath.polar(complex(col - center[0], row - center[1]))
            ori_temp = polar_temp[1]
            if ori_temp < 0:
                ori_temp += pi * 2
            if ori_temp > ori_first and ori_temp < ori_last:
                mask_out[row, col] = 1
    return mask_out

def dsm_roof_cal(roof_type, center, radius, z_center, z_edge, radius_sphere, mask):
    (m, n) = np.shape(mask)
    dsm_out = np.zeros((m, n), dtype=float)
    if roof_type == 0:
        dsm_out = mask * z_edge
    elif roof_type == 1:
        z_h = z_center - z_edge
        for row in range(m):
            for col in range(n):
                dis = sqrt((row - center[1]) ** 2 + (col - center[0]) ** 2)
                dsm_out[row, col] = z_center - z_h * dis / radius
        dsm_out = mask * dsm_out
    elif roof_type == 2:
        for row in range(m):
            for col in range(n):
                dis = sqrt((row - center[1]) ** 2 + (col - center[0]) ** 2)
                if radius_sphere < dis:
                    continue
                height_ht = sqrt(radius_sphere ** 2 - dis ** 2)
                #sita_sp = asin(dis / radius_sphere)
                dsm_out[row, col] = z_center + height_ht - radius_sphere
        dsm_out = mask * dsm_out

    return dsm_out

def circle_dsm(circle_radian, circle_roof, label_mask, dsm):
    pad = 10
    padimg = 100
    imgsize = np.shape(label_mask)
    dsm_out = dsm.copy()
    for cl, circle_l in enumerate(circle_radian):
        lnum = circle_l[0][0]
        img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
        loc_mask_xy = np.where(label_mask == lnum)
        min_locy = np.min(loc_mask_xy[0])
        max_locy = np.max(loc_mask_xy[0])
        min_locx = np.min(loc_mask_xy[1])
        max_locx = np.max(loc_mask_xy[1])
        img_mask[loc_mask_xy] = 1
        img_mask_loc = img_mask[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        dsm_loc = dsm[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad]
        xy_loc2org = np.array([min_locx - pad, min_locy - pad])
        roof_para_con = circle_roof[cl]
        if len(roof_para_con) < 1:
            continue
        for circle in circle_l:
            center = circle[1]
            for connum, concircle in enumerate(circle[3]):
                dsm_temp = np.zeros((imgsize[0], imgsize[1]), dtype=float)
                mask_temp = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
                radius = circle[2][connum]
                radian_first, radian_last, radian = concircle[1] - xy_loc2org, concircle[2] - xy_loc2org, concircle[0]
                roof_para = roof_para_con[connum]
                z_edge, z_center = roof_para[4], roof_para[3]
                # mask awareness
                circle_mask = img_mask_loc * 0
                center_loc = center - xy_loc2org
                circle_mask = cv2.circle(circle_mask, (round(center_loc[0]), round(center_loc[1])),
                                         round(radius), 1.0, -1)
                # notice the axis direction of polar coordinates
                polar_first = cmath.polar(complex(radian_first[0] - center_loc[0], radian_first[1] - center_loc[1]))
                polar_last = cmath.polar(complex(radian_last[0] - center_loc[0], radian_last[1] - center_loc[1]))
                ori_first, ori_last = polar_first[1], polar_last[1]
                if ori_first < 0:
                    ori_first += pi * 2
                if ori_last < ori_first:
                    ori_last += pi * 2

                # if ori_last == ori_first, this is a full circle, otherwise, it is a circle with radian
                circle_mask_cover = circle_mask.copy()
                if ori_last != ori_first:
                    circle_mask_cover = radian_mask(center_loc, radius, ori_first, ori_last, circle_mask)
                #calculate DSM file
                dsm_roof = dsm_roof_cal(roof_para[2], center_loc, radius,
                                        z_center, z_edge, roof_para[5], circle_mask_cover)
                mask_temp[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad] = circle_mask_cover
                dsm_temp[min_locy - pad:max_locy + pad, min_locx - pad:max_locx + pad] = dsm_roof
                dsm_out[np.where(mask_temp)] = dsm_temp[np.where(mask_temp)]

    return dsm_out


def circle_obj(circle_radian, circle_roof, label_mask, min_height):
    pad = 10
    padimg = 100
    Th_ang = pi/18
    imgsize = np.shape(label_mask)
    m = imgsize[0] - padimg * 2
    n = imgsize[1] - padimg * 2
    z0 = min_height
    node = []
    face = []
    texture = []
    ndm = 1
    for cl, circle_l in enumerate(circle_radian):
        roof_para_con = circle_roof[cl]
        if len(roof_para_con) < 1:
            continue
        for circle in circle_l:
            center = circle[1]
            center_org = center - padimg
            for connum, concircle in enumerate(circle[3]):
                radius = circle[2][connum]
                radian_first, radian_last, radian = concircle[1], concircle[2], concircle[0]
                roof_para = roof_para_con[connum]
                z_edge, z_center = roof_para[4], roof_para[3]
                #notice the axis direction of polar coordinates
                polar_first = cmath.polar(complex(radian_first[0] - center[0], radian_first[1] - center[1]))
                polar_last = cmath.polar(complex(radian_last[0] - center[0], radian_last[1] - center[1]))
                ori_first, ori_last = polar_first[1], polar_last[1]
                if ori_first < 0:
                    ori_first += pi * 2
                if ori_last < ori_first:
                    ori_last += pi * 2

                #define circle node
                num_node = round(2 * pi / Th_ang)
                tempn = []  # node
                tempvt = []  # vector
                tempf = []  # face (triangle)
                # if ori_last == ori_first, this is a full circle, otherwise, it is a circle with radian
                if ori_last != ori_first:
                    num_node = round((ori_last - ori_first) / Th_ang) + 1

                # add basic plane
                # base center node
                tempn.append(np.array([center_org[0], center_org[1], z0]))
                tempvt.append(np.array([center_org[0] / n, center_org[1] / m]))
                # basic plane
                if ori_last == ori_first:
                    for i_node in range(num_node):
                        sita = Th_ang * i_node
                        node_x = center_org[0] + cos(sita) * radius
                        node_y = center_org[1] + sin(sita) * radius
                        tempn.append(np.array([node_x, node_y, z0]))
                        tempvt.append(np.array([node_x / n, node_y / m]))
                        if i_node == num_node - 1:
                            tempf.append(np.array([ndm, ndm + 1, ndm + i_node + 1]))
                        else:
                            tempf.append(np.array([ndm, ndm + i_node + 2, ndm + i_node + 1]))
                    ndm += num_node + 1
                else:
                    Th_ang1 = (ori_last - ori_first) / (num_node - 1)
                    for i_node in range(num_node):
                        sita = Th_ang1 * i_node + ori_first
                        node_x = center_org[0] + cos(sita) * radius
                        node_y = center_org[1] + sin(sita) * radius
                        tempn.append(np.array([node_x, node_y, z0]))
                        tempvt.append(np.array([node_x / n, node_y / m]))
                        if i_node < num_node - 1:
                            tempf.append(np.array([ndm, ndm + i_node + 2, ndm + i_node + 1]))
                    ndm += num_node + 1

                # add body cylinder
                if ori_last == ori_first:
                    for i_node in range(num_node):
                        sita = Th_ang * i_node
                        node_x = center_org[0] + cos(sita) * radius
                        node_y = center_org[1] + sin(sita) * radius
                        tempn.append(np.array([node_x, node_y, z_edge]))
                        tempvt.append(np.array([node_x / n, node_y / m]))

                        if i_node == num_node - 1:
                            tempf.append(np.array([ndm - num_node + i_node,
                                                   ndm - num_node, ndm + i_node]))
                            tempf.append(np.array([ndm - num_node, ndm, ndm + i_node]))
                        else:
                            tempf.append(np.array([ndm - num_node + i_node,
                                                   ndm - num_node + i_node + 1, ndm + i_node]))
                            tempf.append(np.array([ndm - num_node + i_node + 1,
                                                   ndm + i_node + 1, ndm + i_node]))
                    ndm += num_node
                else:
                    Th_ang1 = (ori_last - ori_first) / (num_node - 1)
                    for i_node in range(num_node):
                        sita = Th_ang1 * i_node + ori_first
                        node_x = center_org[0] + cos(sita) * radius
                        node_y = center_org[1] + sin(sita) * radius
                        tempn.append(np.array([node_x, node_y, z_edge]))
                        tempvt.append(np.array([node_x / n, node_y / m]))
                        if i_node < num_node - 1:
                            tempf.append(np.array([ndm - num_node + i_node,
                                                   ndm - num_node + i_node + 1, ndm + i_node]))
                            tempf.append(np.array([ndm - num_node + i_node + 1,
                                                   ndm + i_node + 1, ndm + i_node]))
                    ndm += num_node

                # add roof structure
                # 1. flat roof & 2. cone roof
                if roof_para[2] < 2:
                    tempn.append(np.array([center_org[0], center_org[1], z_center]))
                    tempvt.append(np.array([center_org[0] / n, center_org[1] / m]))
                    ndm1 = ndm - num_node - 1
                    if ori_last == ori_first:
                        for i_node in range(num_node):
                            if i_node == num_node - 1:
                                tempf.append(np.array([ndm, ndm1 + i_node + 1, ndm1 + 1]))
                            else:
                                tempf.append(np.array([ndm, ndm1 + i_node + 1, ndm1 + i_node + 2]))
                        ndm += 1
                    else:
                        # add side faces (for radian)
                        tempf.append(np.array([ndm1 - num_node, ndm1 + 1, ndm]))
                        tempf.append(np.array([ndm1 - num_node, ndm1 - num_node + 1, ndm1 + 1]))
                        tempf.append(np.array([ndm1 - num_node, ndm - 1, ndm1]))
                        tempf.append(np.array([ndm1 - num_node, ndm, ndm - 1]))
                        for i_node in range(num_node):
                            if i_node < num_node - 1:
                                tempf.append(np.array([ndm, ndm1 + i_node + 1, ndm1 + i_node + 2]))
                        ndm += 1
                # 3. sphere roof
                # use 4 layer to represent the spherical shape
                else:
                    r_sp = roof_para[5]
                    sita_sp = asin(radius/r_sp)
                    if ori_last == ori_first:
                        # layer 1 to 3
                        for i_layer in range(3):
                            sita1 = (3 - i_layer) * sita_sp / 4
                            for i_node in range(num_node):
                                sita = Th_ang * i_node
                                node_x = center_org[0] + cos(sita) * sin(sita1) * r_sp
                                node_y = center_org[1] + sin(sita) * sin(sita1) * r_sp
                                z_layer = z_center + (cos(sita1) - 1) * r_sp
                                tempn.append(np.array([node_x, node_y, z_layer]))
                                tempvt.append(np.array([node_x / n, node_y / m]))
                                if i_node == num_node - 1:
                                    tempf.append(np.array([ndm - 1, ndm - num_node, ndm + num_node - 1]))
                                    tempf.append(np.array([ndm - num_node, ndm, ndm + num_node - 1]))
                                else:
                                    tempf.append(np.array([ndm - num_node + i_node,
                                                           ndm - num_node + i_node + 1, ndm + i_node]))
                                    tempf.append(np.array([ndm - num_node + i_node + 1,
                                                           ndm + i_node + 1, ndm + i_node]))
                            ndm += num_node
                        #layer 4
                        tempn.append(np.array([center_org[0], center_org[1], z_center]))
                        tempvt.append(np.array([center_org[0] / n, center_org[1] / m]))
                        ndm1 = ndm - num_node - 1
                        for i_node in range(num_node):
                            if i_node == num_node - 1:
                                tempf.append(np.array([ndm, ndm1 + i_node + 1, ndm1 + 1]))
                            else:
                                tempf.append(np.array([ndm, ndm1 + i_node + 1, ndm1 + i_node + 2]))
                        ndm += 1

                    else:
                        Th_ang1 = (ori_last - ori_first) / (num_node - 1)
                        # add side faces (for radian)
                        ndm2 = ndm - num_node - 1
                        tempf.append(np.array([ndm2 - num_node, ndm2 - num_node + 1, ndm2 + 1]))
                        tempf.append(np.array([ndm2 - num_node, ndm - 1, ndm2]))
                        # layer 1 to 3
                        for i_layer in range(3):
                            sita1 = (3 - i_layer) * sita_sp / 4
                            for i_node in range(num_node):
                                sita = Th_ang1 * i_node + ori_first
                                node_x = center_org[0] + cos(sita) * sin(sita1) * r_sp
                                node_y = center_org[1] + sin(sita) * sin(sita1) * r_sp
                                z_layer = z_center + (cos(sita1) - 1) * r_sp
                                tempn.append(np.array([node_x, node_y, z_layer]))
                                tempvt.append(np.array([node_x / n, node_y / m]))
                                if i_node < num_node - 1:
                                    tempf.append(np.array([ndm - num_node + i_node,
                                                           ndm - num_node + i_node + 1, ndm + i_node]))
                                    tempf.append(np.array([ndm - num_node + i_node + 1,
                                                           ndm + i_node + 1, ndm + i_node]))
                            ndm += num_node
                            tempf.append(np.array([ndm2 - num_node, ndm - num_node, ndm]))
                            tempf.append(np.array([ndm2 - num_node, ndm - 1, ndm - num_node - 1]))
                        # layer 4
                        tempn.append(np.array([center_org[0], center_org[1], z_center]))
                        tempvt.append(np.array([center_org[0] / n, center_org[1] / m]))
                        ndm1 = ndm - num_node - 1
                        for i_node in range(num_node):
                            if i_node < num_node - 1:
                                tempf.append(np.array([ndm, ndm1 + i_node + 1, ndm1 + i_node + 2]))
                        tempf.append(np.array([ndm2 - num_node, ndm2 + 1, ndm]))
                        tempf.append(np.array([ndm2 - num_node, ndm, ndm - 1]))
                        ndm += 1
                if len(tempn) < 1:
                    continue
                tempn = np.array(tempn)
                tempn[:, 1] = m - tempn[:, 1]
                tempvt = np.array(tempvt)
                tempvt[:, 1] = 1 - tempvt[:, 1]
                tempf = np.array(tempf)
                tempf = tempf[:,[2,1,0]]
                node.extend(tempn)
                face.extend(tempf)
                texture.extend(tempvt)
    return node, face, texture
