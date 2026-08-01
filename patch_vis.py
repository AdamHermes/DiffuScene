with open('/Users/lehoangan/Documents/GitHub/ROOM/echoscene/scripts/rendering/vis_json_compare.py', 'r') as f:
    content = f.read()

import re

# We will patch plot_scene to draw a room boundary using limits if no floor object exists
new_plot_scene = """
def plot_scene(ax, objects, title, limits, no_title=False, sideways=False):
    ax.clear()
    has_floor = False
    for obj in objects:
        if obj["name"] == "floor":
            has_floor = True
            break
            
    # Draw limits as room boundary if no floor exists
    if not has_floor:
        min_x, max_x, min_z, max_z = limits
        room_boundary = patches.Rectangle(
            (min_x, min_z), max_x - min_x, max_z - min_z,
            linewidth=2.5, edgecolor='red', facecolor='none', alpha=0.9
        )
        ax.add_patch(room_boundary)
        
    for obj in objects:
        obb_corners = get_obb_corners(obj["x"], obj["z"], obj["l"], obj["w"], obj["angle"])
        
        if sideways:
            obb_corners = np.stack([-obb_corners[:, 1], obb_corners[:, 0]], axis=1)
            cx, cz = -obj["z"], obj["x"]
        else:
            cx, cz = obj["x"], obj["z"]
        
        min_x, min_z = np.min(obb_corners, axis=0)
        max_x, max_z = np.max(obb_corners, axis=0)
        
        if obj["name"] == "floor":
            # Draw solid red boundary for the room
            room_boundary = patches.Polygon(
                obb_corners, closed=True, facecolor='none', 
                edgecolor='red', linewidth=2.5, alpha=0.9
            )
            ax.add_patch(room_boundary)
        else:
            obb_polygon = patches.Polygon(
                obb_corners, closed=True, facecolor=obj["color"], 
                edgecolor='black', alpha=0.7, linewidth=1.5
            )
            ax.add_patch(obb_polygon)
            
            # Draw AABB (Dashed Red Line)
            aabb_rect = patches.Rectangle(
                (min_x, min_z), max_x - min_x, max_z - min_z, 
                linewidth=1, edgecolor='#e74c3c', facecolor='none', 
                linestyle='--', alpha=0.6
            )
            ax.add_patch(aabb_rect)
            
            # Draw Object Label
            ax.text(cx, cz, obj["name"], ha='center', va='center', 
                    fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_aspect('equal')
    ax.set_xlim(limits[0] - 0.5, limits[1] + 0.5)
    ax.set_ylim(limits[2] - 0.5, limits[3] + 0.5)
    if not no_title:
        ax.set_title(title, fontsize=10)
    ax.axis('off')
"""

content = re.sub(
    r"def plot_scene\(ax, objects, title, limits, no_title=False, sideways=False\):.*?ax\.axis\('off'\)", 
    new_plot_scene.strip(), 
    content, 
    flags=re.DOTALL
)

with open('/Users/lehoangan/Documents/GitHub/ROOM/echoscene/scripts/rendering/vis_json_compare.py', 'w') as f:
    f.write(content)
