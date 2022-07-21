from skimage import io,data,filters,segmentation,measure,morphology,color,feature,draw
from skimage.util import img_as_ubyte
from scipy import ndimage,optimize,signal
import trimesh
import numpy as np
from math import pi,sqrt,sin,cos,ceil,floor,degrees
from building_modelfit import *
import cv2
import json
import copy

def extract_ridge_dir1(ndm):
    tempf=[]
    tempf.extend(np.array([[ndm+3,ndm+2,ndm],[ndm,ndm+1,ndm+3],[ndm+5,ndm+1,ndm],[ndm,ndm+4,ndm+5],[ndm+4,ndm+8,ndm+5]]))
    tempf.extend(np.array([[ndm+7,ndm+3,ndm+1],[ndm+1,ndm+5,ndm+7],[ndm+6,ndm+2,ndm+3],[ndm+3,ndm+7,ndm+6],[ndm+7,ndm+9,ndm+6]]))
    tempf.extend(np.array([[ndm+4,ndm,ndm+2],[ndm+2,ndm+6,ndm+4]]))
    tempf.extend(np.array([[ndm+9,ndm+8,ndm+4],[ndm+4,ndm+6,ndm+9],[ndm+7,ndm+5,ndm+8],[ndm+8,ndm+9,ndm+7]]))
    return tempf

def extract_ridge_dir2(ndm):
    tempf=[]
    tempf.extend(np.array([[ndm+3,ndm+2,ndm],[ndm,ndm+1,ndm+3],[ndm+5,ndm+1,ndm],[ndm,ndm+4,ndm+5]]))
    tempf.extend(np.array([[ndm+7,ndm+3,ndm+1],[ndm+1,ndm+5,ndm+7],[ndm+5,ndm+9,ndm+7],[ndm+6,ndm+2,ndm+3],[ndm+3,ndm+7,ndm+6]]))
    tempf.extend(np.array([[ndm+4,ndm,ndm+2],[ndm+2,ndm+6,ndm+4],[ndm+6,ndm+8,ndm+4]]))
    tempf.extend(np.array([[ndm+8,ndm+9,ndm+4],[ndm+5,ndm+4,ndm+9],[ndm+7,ndm+8,ndm+6],[ndm+9,ndm+8,ndm+7]]))
    return tempf

def cal_node_vt_ridge(rec,gxy,zv0,zva,m,n):
    tempn=[]
    tempvt=[]
    tempn.extend(np.vstack((rec[8::2],rec[9::2],np.ones((4,),dtype=float)*zv0)).T)
    tempn.extend(np.vstack((rec[8::2],rec[9::2],np.ones((4,),dtype=float)*zva[1])).T)
    tempn.extend(np.array([[gxy[0],gxy[1],zva[2]],[gxy[2],gxy[3],zva[2]]]))
    tempvt.extend(np.vstack((rec[8::2]/n,rec[9::2]/m)).T)
    tempvt.extend(np.vstack((rec[8::2]/n,rec[9::2]/m)).T)
    tempvt.extend(np.array([[gxy[0]/n,gxy[1]/m],[gxy[2]/n,gxy[3]/m]]))
    return tempn,tempvt

def cal_obj(shape2d,para3d,img_dsm,min_height):
    p2d = copy.deepcopy(shape2d)
    p2d = p2d - [0, 0, 100, 100, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100]
    for i, rec in enumerate(p2d):
        rec[np.where(rec[8:] < 1)[0] + 8] = 1
    loc2d = copy.deepcopy(p2d[:, 8:])
    loc2d[np.where(loc2d < 1)] = 1
    loc2dm = loc2d[:, 1::2]
    loc2dn = loc2d[:, 0::2]
    (m, n) = np.shape(img_dsm)
    loc2dm[np.where(loc2dm > m)] = m
    loc2dn[np.where(loc2dn > n)] = n

    p2dt = np.vstack((p2d[:, 0:8].T, loc2dn[:, 0], loc2dm[:, 0], loc2dn[:, 1], loc2dm[:, 1], \
                      loc2dn[:, 2], loc2dm[:, 2], loc2dn[:, 3], loc2dm[:, 3])).T
    p3d = copy.deepcopy(para3d)

    zv0 = min_height
    node = []
    face = []
    texture = []
    ndm = 1
    for i in range(len(p2d)):
        rec = p2dt[i, :]
        zva = p3d[i]
        tempn = []  # node
        tempvt = []  # vector
        tempf = []  # face (triangle)

        # for flat, 8 nodes, 12 faces
        if zva[0] == 1:
            tempn.extend(np.vstack((rec[8::2], rec[9::2], np.ones((4,), dtype=float) * zv0)).T)
            tempn.extend(np.vstack((rec[8::2], rec[9::2], np.ones((4,), dtype=float) * zva[1])).T)
            tempvt.extend(np.vstack((rec[8::2] / n, rec[9::2] / m)).T)
            tempvt.extend(np.vstack((rec[8::2] / n, rec[9::2] / m)).T)
            tempf.extend(np.array(
                [[ndm + 3, ndm + 2, ndm], [ndm, ndm + 1, ndm + 3], [ndm + 5, ndm + 1, ndm], [ndm, ndm + 4, ndm + 5]]))
            tempf.extend(np.array(
                [[ndm + 7, ndm + 3, ndm + 1], [ndm + 1, ndm + 5, ndm + 7], [ndm + 6, ndm + 2, ndm + 3],
                 [ndm + 3, ndm + 7, ndm + 6]]))
            tempf.extend(np.array([[ndm + 4, ndm, ndm + 2], [ndm + 2, ndm + 6, ndm + 4], [ndm + 7, ndm + 5, ndm + 4],
                                   [ndm + 4, ndm + 6, ndm + 7]]))
            ndm += 8

        # for gable, 10 nodes, 16 faces
        elif zva[0] == 2:
            if zva[5] == 1:
                gxyp = np.array(
                    [(rec[8] + rec[10]) / 2, (rec[9] + rec[11]) / 2, (rec[12] + rec[14]) / 2, (rec[13] + rec[15]) / 2])
                tempn0, tempvt0 = cal_node_vt_ridge(rec, gxyp, zv0, zva, m, n)
                tempf0 = extract_ridge_dir1(ndm)
                tempn.extend(tempn0)
                tempvt.extend(tempvt0)
                tempf.extend(tempf0)
            elif zva[5] == 2:
                gxyv = np.array(
                    [(rec[8] + rec[12]) / 2, (rec[9] + rec[13]) / 2, (rec[10] + rec[14]) / 2, (rec[11] + rec[15]) / 2])
                tempn0, tempvt0 = cal_node_vt_ridge(rec, gxyv, zv0, zva, m, n)
                tempf0 = extract_ridge_dir2(ndm)
                tempn.extend(tempn0)
                tempvt.extend(tempvt0)
                tempf.extend(tempf0)
            ndm += 10

        # for hip, 10 nodes, 16 faces
        elif zva[0] == 3:
            if zva[5] == 1:
                rt = zva[3] / rec[4]
                gxy0 = np.array(
                    [(rec[8] + rec[10]) / 2, (rec[9] + rec[11]) / 2, (rec[12] + rec[14]) / 2, (rec[13] + rec[15]) / 2])
                gxy = np.array([gxy0[0] * (1 - rt) + gxy0[2] * rt, gxy0[1] * (1 - rt) + gxy0[3] * rt, \
                                gxy0[0] * rt + gxy0[2] * (1 - rt), gxy0[1] * rt + gxy0[3] * (1 - rt)])
                tempn0, tempvt0 = cal_node_vt_ridge(rec, gxy, zv0, zva, m, n)
                tempf0 = extract_ridge_dir1(ndm)
                tempn.extend(tempn0)
                tempvt.extend(tempvt0)
                tempf.extend(tempf0)
            elif zva[5] == 2:
                rt = zva[3] / rec[5]
                gxy0 = np.array(
                    [(rec[8] + rec[12]) / 2, (rec[9] + rec[13]) / 2, (rec[10] + rec[14]) / 2, (rec[11] + rec[15]) / 2])
                gxy = np.array([gxy0[0] * (1 - rt) + gxy0[2] * rt, gxy0[1] * (1 - rt) + gxy0[3] * rt, \
                                gxy0[0] * rt + gxy0[2] * (1 - rt), gxy0[1] * rt + gxy0[3] * (1 - rt)])
                tempn0, tempvt0 = cal_node_vt_ridge(rec, gxy, zv0, zva, m, n)
                tempf0 = extract_ridge_dir2(ndm)
                tempn.extend(tempn0)
                tempvt.extend(tempvt0)
                tempf.extend(tempf0)
            ndm += 10

        # for pyramid, 9 nodes, 14 faces
        elif zva[0] == 4:
            gxy = np.array([np.mean(rec[8::2]), np.mean(rec[9::2])])
            tempn.extend(np.vstack((rec[8::2], rec[9::2], np.ones((4,), dtype=float) * zv0)).T)
            tempn.extend(np.vstack((rec[8::2], rec[9::2], np.ones((4,), dtype=float) * zva[1])).T)
            tempn.extend(np.array([gxy[0], gxy[1], zva[2]]))
            tempvt.extend(np.vstack((rec[8::2] / n, rec[9::2] / m)).T)
            tempvt.extend(np.vstack((rec[8::2] / n, rec[9::2] / m)).T)
            tempvt.extend(np.array([gxy[0] / n, gxy[1] / m]))
            tempf.extend(np.array(
                [[ndm + 3, ndm + 2, ndm], [ndm, ndm + 1, ndm + 3], [ndm + 5, ndm + 1, ndm], [ndm, ndm + 4, ndm + 5]]))
            tempf.extend(np.array(
                [[ndm + 7, ndm + 3, ndm + 1], [ndm + 1, ndm + 5, ndm + 7], [ndm + 6, ndm + 2, ndm + 3],
                 [ndm + 3, ndm + 7, ndm + 6]]))
            tempf.extend(np.array([[ndm + 4, ndm, ndm + 2], [ndm + 2, ndm + 6, ndm + 4]]))
            tempf.extend(np.array(
                [[ndm + 8, ndm + 5, ndm + 4], [ndm + 8, ndm + 7, ndm + 5], [ndm + 8, ndm + 6, ndm + 7],
                 [ndm + 8, ndm + 4, ndm + 6]]))
            ndm += 9

        # for pyramid, 12 nodes, 20 faces
        elif zva[0] == 5:
            rt1 = zva[3] / rec[4]
            rt2 = zva[3] / rec[5]
            gxy = np.array([rec[8] * (1 - rt1) * (1 - rt2) + rec[10] * rt1 * (1 - rt2) + rec[12] * (1 - rt1) * rt2 +
                            rec[14] * rt1 * rt2, \
                            rec[9] * (1 - rt1) * (1 - rt2) + rec[11] * rt1 * (1 - rt2) + rec[13] * (1 - rt1) * rt2 +
                            rec[15] * rt1 * rt2, \
                            rec[8] * rt1 * (1 - rt2) + rec[10] * (1 - rt1) * (1 - rt2) + rec[12] * rt1 * rt2 + rec[
                                14] * (1 - rt1) * rt2, \
                            rec[9] * rt1 * (1 - rt2) + rec[11] * (1 - rt1) * (1 - rt2) + rec[13] * rt1 * rt2 + rec[
                                15] * (1 - rt1) * rt2, \
                            rec[8] * (1 - rt1) * rt2 + rec[10] * rt1 * rt2 + rec[12] * (1 - rt1) * (1 - rt2) + rec[
                                14] * rt1 * (1 - rt2), \
                            rec[9] * (1 - rt1) * rt2 + rec[11] * rt1 * rt2 + rec[13] * (1 - rt1) * (1 - rt2) + rec[
                                15] * rt1 * (1 - rt2), \
                            rec[8] * rt1 * rt2 + rec[10] * (1 - rt1) * rt2 + rec[12] * rt1 * (1 - rt2) + rec[14] * (
                                        1 - rt1) * (1 - rt2), \
                            rec[9] * rt1 * rt2 + rec[11] * (1 - rt1) * rt2 + rec[13] * rt1 * (1 - rt2) + rec[15] * (
                                        1 - rt1) * (1 - rt2)])
            tempn.extend(np.vstack((rec[8::2], rec[9::2], np.ones((4,), dtype=float) * zv0)).T)
            tempn.extend(np.vstack((rec[8::2], rec[9::2], np.ones((4,), dtype=float) * zva[1])).T)
            tempn.extend(np.array([[gxy[0], gxy[1], zva[2]], [gxy[2], gxy[3], zva[2]], [gxy[4], gxy[5], zva[2]],
                                   [gxy[6], gxy[7], zva[2]]]))
            tempvt.extend(np.vstack((rec[8::2] / n, rec[9::2] / m)).T)
            tempvt.extend(np.vstack((rec[8::2] / n, rec[9::2] / m)).T)
            tempvt.extend(np.vstack((gxy[0::2] / n, gxy[1::2] / m)).T)
            tempf.extend(np.array(
                [[ndm + 3, ndm + 2, ndm], [ndm, ndm + 1, ndm + 3], [ndm + 5, ndm + 1, ndm], [ndm, ndm + 4, ndm + 5]]))
            tempf.extend(np.array(
                [[ndm + 7, ndm + 3, ndm + 1], [ndm + 1, ndm + 5, ndm + 7], [ndm + 6, ndm + 2, ndm + 3],
                 [ndm + 3, ndm + 7, ndm + 6]]))
            tempf.extend(np.array([[ndm + 4, ndm, ndm + 2], [ndm + 2, ndm + 6, ndm + 4], [ndm + 10, ndm + 9, ndm + 8],
                                   [ndm + 11, ndm + 9, ndm + 10]]))
            tempf.extend(np.array(
                [[ndm + 8, ndm + 9, ndm + 4], [ndm + 4, ndm + 9, ndm + 5], [ndm + 11, ndm + 7, ndm + 5],
                 [ndm + 5, ndm + 9, ndm + 11]]))
            tempf.extend(np.array(
                [[ndm + 11, ndm + 6, ndm + 7], [ndm + 6, ndm + 11, ndm + 10], [ndm + 8, ndm + 4, ndm + 6],
                 [ndm + 6, ndm + 10, ndm + 8]]))
            ndm += 12
        tempn = np.array(tempn)
        tempn[:, 1] = m - tempn[:, 1]
        tempvt = np.array(tempvt)
        tempvt[:, 1] = 1 - tempvt[:, 1]
        node.extend(tempn)
        face.extend(tempf)
        texture.extend(tempvt)
    return node,face,texture

#deal with irregular building
def DSMtoMesh(decp_ir, img_dsm_t, L_mask, min_height):
    imgsize_t = np.shape(img_dsm_t)
    dsm0 = img_dsm_t[100:imgsize_t[0] - 100, 100:imgsize_t[1] - 100]
    imgsize0 = np.shape(dsm0)

    irr_n = []
    irr_vt = []
    irr_f = []

    for pt_ir in decp_ir:
        pt_temp = np.array(pt_ir)
        pt_row = pt_temp[:, 2] - 100
        pt_col = pt_temp[:, 1] - 100
        mask = draw.polygon2mask(imgsize0, np.array([pt_row, pt_col]).T).astype(np.uint8)

        range_local = np.array([min(pt_row) - 10, max(pt_row) + 10, min(pt_col) - 10, max(pt_col) + 10])

        range_local[0] = max(0, range_local[0])
        range_local[1] = min(imgsize0[0], range_local[1])
        range_local[2] = max(0, range_local[2])
        range_local[3] = min(imgsize0[1], range_local[3])

        mask_loc = mask[range_local[0]:range_local[1], range_local[2]:range_local[3]]

        kernel_len = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, kernel_len))
        mask_edge = (cv2.dilate(mask_loc, kernel)) - mask_loc
        dsm_loc = mask_loc * dsm0[range_local[0]:range_local[1], range_local[2]:range_local[3]]
        # dsm_loc[np.where(dsm_loc<min_height)]=min_height
        dsm_loc = dsm_loc + mask_edge * min_height

        rangey = range_local[1] - range_local[0]
        rangex = range_local[3] - range_local[2]

        node_list = []
        vt_list = []
        face_list = []

        # construct grid mesh
        for row in range(rangey):
            for col in range(rangex):
                node_xyz = [col + range_local[2], row + range_local[0], dsm_loc[row, col]]
                node_list.append(np.array(node_xyz))
                vt_list.append(np.array([(col + range_local[2]) / imgsize0[1], (row + range_local[1]) / imgsize0[0]]))
                node_id = row * (rangex) + col + 1
                if row < rangey - 1 and col < rangex - 1:
                    face_list.append(np.array([node_id, node_id + 1, node_id + rangex]))
                    face_list.append(np.array([node_id + 1, node_id + 1 + rangex, node_id + rangex]))

        # mask out non-building face
        for fid, tempf in enumerate(face_list):
            face_height = node_list[tempf[0] - 1][2] + node_list[tempf[1] - 1][2] \
                          + node_list[tempf[2] - 1][2]
            if face_height <= 3 * min_height:
                face_list[fid] = []

        face_list_mask = [x for x in copy.deepcopy(face_list) if x != []]

        # relabel node
        node_re = np.zeros((len(node_list),), dtype=np.int32)
        for tempf in face_list_mask:
            node_re[tempf[0] - 1] = 1
            node_re[tempf[1] - 1] = 1
            node_re[tempf[2] - 1] = 1
        node_re0 = np.vstack((range(len(node_re)), range(len(node_re)))).T
        node_flag = 1
        for nid, tempn in enumerate(node_list):
            if node_re[nid] == 0:
                node_list[nid] = []
                vt_list[nid] = []
                node_re0[nid, 1] = 0
            else:
                node_re[nid] = nid
                node_re0[nid, 1] = node_flag
                node_flag += 1
        node_list_mask = [x for x in node_list if x != []]
        vt_list_mask = [x for x in vt_list if x != []]

        # relable face
        face_list_mask_new = copy.deepcopy(face_list_mask)
        for fid, tempfnew in enumerate(face_list_mask_new):
            for tid in range(3):
                node_id = tempfnew[tid] - 1
                node_id_new = node_re0[node_id, 1] - 1
                face_list_mask_new[fid][tid] = node_id_new + 1

        node_mask = np.array(node_list_mask)
        face_mask = np.array(face_list_mask_new).astype(np.uint32) - 1
        vt_mask = np.array(vt_list_mask)

        mesh_vis = trimesh.visual.TextureVisuals(uv=vt_mask)
        mesh_obj = trimesh.Trimesh(vertices=node_mask, faces=face_mask, visual=mesh_vis)
        mesh_simp = mesh_obj.simplify_quadratic_decimation(1000)

        node_simp = mesh_simp.vertices
        face_simp = mesh_simp.faces
        vt_simp = []

        for nid in range(len(node_simp)):
            node_simp[nid, 1] = imgsize0[0] - node_simp[nid, 1]
            vt_simp.append(np.array([node_simp[nid, 0] / imgsize0[1], node_simp[nid, 1] / imgsize0[0]]))

        irr_n.append(node_simp)
        irr_vt.append(vt_simp)
        irr_f.append(face_simp)

    return irr_n, irr_f, irr_vt

def MeshMerge_ir(node_r, face_r, texture_r, node_ir, face_ir, texture_ir):
    node_merge = copy.deepcopy(node_r)
    face_merge = copy.deepcopy(face_r)
    vt_merge = copy.deepcopy(texture_r)

    node_num = len(node_r)
    for irr in range(len(node_ir)):
        face_ir[irr] += node_num + 1
        node_num += len(node_ir[irr])
        node_merge.extend(node_ir[irr])
        face_merge.extend(face_ir[irr])
        vt_merge.extend(texture_ir[irr])

    return node_merge, face_merge, vt_merge

def MeshMerge(node_r, face_r, texture_r, node_c, face_c, texture_c):
    node_merge = copy.deepcopy(node_r)
    face_merge = copy.deepcopy(face_r)
    vt_merge = copy.deepcopy(texture_r)

    node_num = len(node_r)
    face_c = list(np.array(face_c) + node_num)
    node_merge.extend(node_c)
    face_merge.extend(face_c)
    vt_merge.extend(texture_c)

    return node_merge, face_merge, vt_merge

def write_obj(filename,vertices,faces,texture):
    filemtl = filename.replace('obj','mtl')
    with open(filemtl,'w') as f:
        f.write("newmtl building_model\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 1.000000\n")
        f.write("Ns 0.000000\n")
        f.write("map_Kd model_texture.jpg\n")
    
    minht=np.min(np.array(vertices)[:,2])

    with open(filename,'w') as f:
        f.write("mtllib building_model.mtl\n")
        f.write("usemtl building_model\n")
        for v in vertices:
            if np.isnan(v[2]) == 1:
                v[2] = minht
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))

        f.write("\n")
        for vt in texture:
            f.write("vt %.4f %.4f\n" % (vt[0],vt[1]))
        f.write("\n")
        for face in faces:
            f.write("f %d/%d %d/%d %d/%d\n" % (face[0],face[0],face[1],face[1],face[2],face[2]))


#Display roof type in .obj
def write_obj_roof_type(filename, vertices, faces, texture, shape2d, para3d, decp_ir, ortho):
    p2d = copy.deepcopy(shape2d)
    p2d = p2d - [0, 0, 100, 100, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100]
    imgsize = np.shape(ortho)
    orthoout = copy.deepcopy(ortho)
    ortho_l1 = orthoout[:, :, 0]
    ortho_l2 = orthoout[:, :, 1]
    ortho_l3 = orthoout[:, :, 2]
    roof_color = np.array([[205,40,40],[40,205,40],[40,40,205],[205,205,40],[205,40,205],[40,205,205]])
    for i, rec in enumerate(p2d):
        rec[np.where(rec[8:] < 1)[0] + 8] = 1
        tx = rec[8::2]
        ty = rec[9::2]
        roof_type = int(para3d[i][0])
        mask_rec = fitrec(imgsize[0], imgsize[1], tx, ty)
        ortho_l1[np.where(mask_rec == 1)] = roof_color[roof_type - 1, 0]
        ortho_l2[np.where(mask_rec == 1)] = roof_color[roof_type - 1, 1]
        ortho_l3[np.where(mask_rec == 1)] = roof_color[roof_type - 1, 2]

    if len(decp_ir) > 0:
        for pt_ir in decp_ir:
            pt_temp = np.array(pt_ir)
            pt_row = pt_temp[:, 2] - 100
            pt_col = pt_temp[:, 1] - 100
            mask_ir = draw.polygon2mask(imgsize[0:2], np.array([pt_row, pt_col]).T).astype(np.uint8)
            roof_type = 6
            ortho_l1[np.where(mask_ir == 1)] = roof_color[roof_type - 1, 0]
            ortho_l2[np.where(mask_ir == 1)] = roof_color[roof_type - 1, 1]
            ortho_l3[np.where(mask_ir == 1)] = roof_color[roof_type - 1, 2]

    io.imsave(filename.replace('building_model_roof.obj', 'model_texture_roof.jpg'), orthoout)

    filemtl = filename.replace('obj', 'mtl')
    with open(filemtl, 'w') as f:
        f.write("newmtl building_model_roof\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 1.000000\n")
        f.write("Ns 0.000000\n")
        f.write("map_Kd model_texture_roof.jpg\n")

    minht = np.min(np.array(vertices)[:, 2])

    with open(filename, 'w') as f:
        f.write("mtllib building_model_roof.mtl\n")
        f.write("usemtl building_model_roof\n")
        for v in vertices:
            if np.isnan(v[2]) == 1:
                v[2] = minht
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))

        f.write("\n")
        for vt in texture:
            f.write("vt %.4f %.4f\n" % (vt[0], vt[1]))
        f.write("\n")
        for face in faces:
            f.write("f %d/%d %d/%d %d/%d\n" % (face[0], face[0], face[1], face[1], face[2], face[2]))
        
