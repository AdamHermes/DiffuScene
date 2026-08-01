box_0 = [1.91738921, 2.41268145, 0.38123229, 1.33558154, 1.20901799, 0.98392707]
box_3 = [1.58611517, 0.79358947, 0.46762044, 1.29188275, 0.39794993, -0.00486625]

hy_0 = box_0[1] / 2.0
cy_0 = box_0[4]
min_y_0 = cy_0 - hy_0
max_y_0 = cy_0 + hy_0

hy_3 = box_3[1] / 2.0
cy_3 = box_3[4]
min_y_3 = cy_3 - hy_3
max_y_3 = cy_3 + hy_3

y_overlap = min(max_y_0, max_y_3) - max(min_y_0, min_y_3)
print(f"Object 0 Y: [{min_y_0}, {max_y_0}]")
print(f"Object 3 Y: [{min_y_3}, {max_y_3}]")
print(f"Y Overlap: {y_overlap}")
