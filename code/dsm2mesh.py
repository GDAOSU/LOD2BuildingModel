from scipy import ndimage,optimize,signal
import trimesh
import numpy as np
from math import pi,sqrt,sin,cos,ceil,floor,degrees
from skimage import io,data,filters,segmentation,measure,morphology,color,feature,draw
import cv2
import copy
import os

def DSMtoMesh(L_mask, lnum, img_dsm_t, min_height, tfw):
    imgsize_t = np.shape(img_dsm_t)
    dsm0 = img_dsm_t
    imgsize0 = np.shape(dsm0)

    irr_n = []
    irr_vt = []
    irr_f = []

    img_mask=np.zeros((imgsize[0],imgsize[1]),dtype=np.uint8)
    img_mask[np.where(L_mask==lnum)] = 1
    pt_temp = np.where(img_mask==1)
    pt_row = pt_temp[0]
    pt_col = pt_temp[1]
    mask = img_mask

    range_local = np.array([min(pt_row) - 10, max(pt_row) + 10, min(pt_col) - 10, max(pt_col) + 10])

    range_local[0] = max(0, range_local[0])
    range_local[1] = min(imgsize0[0], range_local[1])
    range_local[2] = max(0, range_local[2])
    range_local[3] = min(imgsize0[1], range_local[3])

    mask_loc = mask[range_local[0]:range_local[1], range_local[2]:range_local[3]]

    kernel_len = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, kernel_len))
    #mask_edge = (cv2.dilate(mask_loc, kernel)) - mask_loc
    mask_edge=0
    dsm_loc = mask_loc * dsm0[range_local[0]:range_local[1], range_local[2]:range_local[3]]
    # dsm_loc[np.where(dsm_loc<min_height)]=min_height
    dsm_loc = dsm_loc + mask_edge * min_height

    rangey = range_local[1] - range_local[0]
    rangex = range_local[3] - range_local[2]

    node_list = []
    vt_list = []
    face_list = []

    # construct grid mesh
    for irow in range(100):
        for icol in range(100):
            col = icol * rangex / 100
            row = irow * rangey / 100
            node_xyz = [col + range_local[2], row + range_local[0], dsm_loc[round(row), round(col)]]
            node_geo = [node_xyz[0] * tfw[0] + tfw[4], -node_xyz[1] * tfw[3] + tfw[5], node_xyz[2]]
            node_list.append(np.array(node_geo))
            vt_list.append(np.array([(col + range_local[2]) / imgsize0[1], (row + range_local[1]) / imgsize0[0]]))
            node_id = irow * 100 + icol + 1
            if irow < 99 and icol < 99:
                face_list.append(np.array([node_id, node_id + 1, node_id + 100]))
                face_list.append(np.array([node_id + 1, node_id + 1 + 100, node_id + 100]))

    # mask out non-building face

    for fid, tempf in enumerate(face_list):
        face_height = node_list[tempf[0] - 1][2] + node_list[tempf[1] - 1][2] \
                      + node_list[tempf[2] - 1][2]
        if face_height <= 3 * min_height and min_height > 0:
            face_list[fid] = []
        if face_height >= 3 * min_height and min_height < 0:
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

    node_simp = np.array(mesh_simp.vertices)
    face_simp = np.array(mesh_simp.faces) + 1
    vt_simp = []

    for nid in range(len(node_simp)):
        node_simp[nid, 1] = imgsize0[0] - node_simp[nid, 1]
        vt_simp.append(np.array([node_simp[nid, 0] / imgsize0[1], node_simp[nid, 1] / imgsize0[0]]))

    #irr_n.append(node_simp)
    #irr_vt.append(vt_simp)
    #irr_f.append(face_simp)
    irr_n.append(np.array(node_mask))
    irr_vt.append(np.array(vt_mask))
    irr_f.append(np.array(face_mask)+1)

    return irr_n, irr_f, irr_vt


def write_obj(filename, vertices, faces, texture,minh):
    filemtl = filename.replace('obj', 'mtl')
    with open(filemtl, 'w') as f:
        f.write("newmtl building_model\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 1.000000\n")
        f.write("Ns 0.000000\n")
        #f.write("map_Kd model_texture.jpg\n")

    minht = minh

    with open(filename, 'w') as f:
        f.write("mtllib building_model.mtl\n")
        f.write("usemtl building_model\n")
        for v in vertices[0]:
        #for v in vertices:
            if v[2] != minht:
                v[2] = minht
            if np.isnan(v[2]):
                v[2] = minht
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))

        f.write("\n")
        #for vt in texture[0]:
        #    f.write("vt %.4f %.4f\n" % (vt[0], vt[1]))
        f.write("\n")
        for face in faces[0]:
            f.write("f %d/%d %d/%d %d/%d\n" % (face[0], face[0], face[1], face[1], face[2], face[2]))

if __name__ == "__main__":
    dsm_path = r'C:\research\other\demtool\test\mesh_ict_usc\odm_meta_dsm_ortho\metashape_mesh_DSM_mod.tif'
    out_path = 'C:/research/other/demtool/test/mesh_ict_usc/odm_meta_dsm_ortho/maskmesh/'
    img_dsm = io.imread(dsm_path)
    imgsize = np.shape(img_dsm)
    img_dsm[np.isnan(img_dsm)] = -100000
    tfw=[0.2000000030,0.0000000000,0.0000000000,-0.2000000030,-153.1603240967,-177.8534698486]

    # exclude small region in building mask
    img_mask = np.zeros((imgsize[0], imgsize[1]), dtype=np.uint8)
    img_mask[np.where(img_dsm > -999)] = 1
    bw_mask = morphology.remove_small_objects(img_mask.astype(bool), min_size=100, connectivity=1)
    bw_mask = ndimage.binary_fill_holes(bw_mask).astype(bool)

    # connectivity detection
    L_mask, L_num = measure.label(bw_mask, return_num=True, connectivity=2)

    node1, face1, texture1 = DSMtoMesh(L_mask, 1, img_dsm, -6.157, tfw)
    filename = os.path.join(out_path, 'mesh1.obj')
    write_obj(filename, node1, face1, texture1, -6.157)

    node2, face2, texture2 = DSMtoMesh(L_mask, 2, img_dsm, 29.076, tfw)
    filename = os.path.join(out_path, 'mesh2.obj')
    write_obj(filename, node2, face2, texture2, 29.076)







