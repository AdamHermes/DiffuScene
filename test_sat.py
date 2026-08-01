import numpy as np

def _obb_corners_xz(cx, cz, l, w, angle_deg):
    angle_rad = np.radians(angle_deg)
    hx, hz = l / 2.0, w / 2.0
    corners = np.array([
        [ hx,  hz],
        [ hx, -hz],
        [-hx, -hz],
        [-hx,  hz]
    ])
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return corners.dot(R.T) + np.array([cx, cz])

def _obb_sat_xz(box_a, angle_a_deg, box_b, angle_b_deg):
    ca_x, ca_z, la, wa = box_a[3], box_a[5], box_a[0], box_a[2]
    cb_x, cb_z, lb, wb = box_b[3], box_b[5], box_b[0], box_b[2]
    corners_a = _obb_corners_xz(ca_x, ca_z, la, wa, angle_a_deg)
    corners_b = _obb_corners_xz(cb_x, cb_z, lb, wb, angle_b_deg)
    
    a_rad = np.radians(angle_a_deg)
    b_rad = np.radians(angle_b_deg)
    axes = [
        np.array([np.cos(a_rad), np.sin(a_rad)]),
        np.array([-np.sin(a_rad), np.cos(a_rad)]),
        np.array([np.cos(b_rad), np.sin(b_rad)]),
        np.array([-np.sin(b_rad), np.cos(b_rad)])
    ]
    
    min_depth = float('inf')
    for axis in axes:
        proj_a = [np.dot(c, axis) for c in corners_a]
        proj_b = [np.dot(c, axis) for c in corners_b]
        min_a, max_a = min(proj_a), max(proj_a)
        min_b, max_b = min(proj_b), max(proj_b)
        
        if max_a <= min_b or max_b <= min_a:
            return False, 0.0
        
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_depth:
            min_depth = overlap
            
    return True, min_depth

# Object 0 after resolution
box_0 = [1.91738921, 2.41268145, 0.38123229, 1.33558154, 1.20901799, 0.98392707]
ang_0 = -89.99251662

# Object 3
box_3 = [1.58611517, 0.79358947, 0.46762044, 1.29188275, 0.39794993, -0.00486625]
ang_3 = -90.02804726

col, depth = _obb_sat_xz(box_0, ang_0, box_3, ang_3)
print(f"Col: {col}, Depth: {depth}")
