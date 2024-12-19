import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math


# Function to calculate perpendicular vectors to a given vector
def perpendicular_vectors(direction):
    # Pick an arbitrary vector that is not parallel to the direction vector
    arbitrary_vector = np.array([1, 0, 0]) if direction[0] == 0 else np.array([0, 1, 0])
    
    # Compute two perpendicular vectors using cross-product
    u = np.cross(direction, arbitrary_vector)
    u /= np.linalg.norm(u)  # Normalize the vector
    v = np.cross(direction, u)
    v /= np.linalg.norm(v)  # Normalize the vector
    return u, v

# Function to generate vertices of the rectangular prism
def generate_prism_vertices(r0, direction, width, height, length):
    # Calculate unit direction vector
    direction = direction / np.linalg.norm(direction)

    # Find perpendicular vectors u and v
    u, v = perpendicular_vectors(direction)

    # Create two planes: base and top rectangle vertices
    half_width, half_height = width / 2, height / 2
    r1 = r0 + length * direction
    
    # Vertices for the base rectangle (at r0)
    base_vertices = np.array([
        r0 + half_width * u + half_height * v,
        r0 - half_width * u + half_height * v,
        r0 - half_width * u - half_height * v,
        r0 + half_width * u - half_height * v
    ])
    
    # Vertices for the top rectangle (at r1)
    top_vertices = np.array([
        r1 + half_width * u + half_height * v,
        r1 - half_width * u + half_height * v,
        r1 - half_width * u - half_height * v,
        r1 + half_width * u - half_height * v
    ])

    return base_vertices, top_vertices

# Plotting function
def plot_prism(base_vertices, top_vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the 6 faces of the rectangular prism
    vertices = np.vstack([base_vertices, top_vertices])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # base
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # side 1
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # side 2
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # side 3
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # side 4
    ]

    # Plot the faces
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    # Set plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Set the range for the axes
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def generate_prisms(prism_vertices_list):
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    vert = []
    for base_vertices, top_vertices in prism_vertices_list:
        # Combine base and top vertices
        vertices = np.vstack([base_vertices, top_vertices])
        
        # Plot the vertices
        vert.extend(vertices)
    return vert
        
 
# Main part of the code
# if __name__ == "__main__":
#     # Line parameters
#     r0 = np.array([0, 0, 0])  # A point on the line (base of the prism)
#     b = []
#     t = []
#     #direction = np.array([1, 2, 3])  # Direction vector of the line
#     #joint_pos = np.array([[1.2670000791549683, 1.3374300003051758, 0.7160000205039978],
#                     #   [1.2723760604858398, 1.465809941291809, 0.7160000205039978],
#                     #   [1.2787532806396484, 1.67618989944458, 0.7160000205039978],
#                     #   [1.2851297855377197, 1.8865699768066406, 0.7159999012947083],
#                     #   [1.2915070056915283, 2.0949997901916504, 0.7159998416900635],
#                     #   [1.2916828393936157, 2.200929641723633, 0.7159998416900635],
#                     #   [1.2918590307235718, 2.3068597316741943, 0.7159998416900635],
#                     #   [1.2918593883514404, 2.368384838104248, 0.7159997820854187],
#                     #   [1.2918606996536255, 2.5183849334716797, 0.7159997224807739]])

#     joint_pos = np.array([[1.2669999599456787, 1.3374298810958862, 0.7160000205039978], [1.27024507522583, 1.465809941291809, 0.7117143869400024], [1.1670883893966675, 1.627822756767273, 0.6256275177001953], [1.0637660026550293, 1.7896671295166016, 0.5394228100776672], [1.2049520015716553, 1.9153385162353516, 0.6274961233139038], [1.274794101715088, 1.979421615600586, 0.6747888326644897], [1.3421106338500977, 1.9388375282287598, 0.6037775278091431]])
#     prism_vertices_list = []
    
        
#     # Plot the rectangular prism
#     #plot_prism(b, t)
#     #print((prisms))
#     plot_prisms(prism_vertices_list)
