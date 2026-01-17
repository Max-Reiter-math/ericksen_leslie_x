import numpy as np
from mpi4py import MPI
import ufl
import basix
from dolfinx.mesh import Mesh, create_mesh, locate_entities_boundary, meshtags
from dolfinx.io import XDMFFile

"""
Creation of strucutred spiral meshes.
Usage example:
    python -m input.spiral_mesh -n_radius 10
"""

def create_spiral_mesh(comm: MPI.Comm, n_ang: int, n_rad: int) -> Mesh:
    """
    Creates a mesh for a disk with radius r2 = 2, with a hole with radius r1 = 1.

    Parameters:
    comm: MPI.comm
    n_ang: int
        mesh resolution along circumference
    n_rad: int
        mesh resolution along radial direction

    Returns:
    domain: dolfinx.mesh.Mesh
        weakly acute triangulation
    """
    n_radius = n_rad+1                          # mesh resolution along radial direction
    if n_ang == 1:
        n_angle  = int( np.ceil(4*np.pi) * (n_rad+1) )  # mesh resolution along circumference
        # NOTE - We choose this to obtain a weakly acute triangulation.
        # circumference for r1 = 1, r2 = 2, is inside 2pi and outside 4pi
        # Accordingly the triangle edge length on the boundary will be:
        #   on the inside: 2pi / n_angle
        #   on the outside: 4pi / n_angle
        # and its height: 1 / n_radius
        # In order to obtain an acute triangulation
        # => 1 / n_radius = 1 / (n+1) > 4pi /n_angle
        # <=> n_angle > 4pi * (n+1)
    else:
        n_angle = n_ang
    
    # Alternating mesh resolution along circumference
    angles      = np.linspace(0, 2*np.pi, n_angle, endpoint = False)
    angles_alt  = angles + np.pi*1/n_angle
    radii       = np.linspace(1,2, n_radius)

    # Point Creation by alternating the angles 
    pts_x   = np.concat( [ np.outer( radii[::2], np.cos(angles)).flatten() , np.outer( radii[1::2], np.cos(angles_alt)).flatten() ])
    pts_y   = np.concat( [ np.outer( radii[::2], np.sin(angles)).flatten() , np.outer( radii[1::2], np.sin(angles_alt)).flatten() ])
    pts     = np.array([ pts_x, pts_y ])

    index_alt = len(radii[::2]) * n_angle # index at which the points with alternating angle resolution begin
    
    # Cell creation
    cells = np.array([[],[],[]])
    # NOTE - this could further be vectorized, but
    # 1.) I am too lazy
    # 2.) it is sufficient for my purposes sinc e.g.:
    #       Mesh creation took 3.1183109283447266 seconds.
    #       Created 525,213 nodes and 1,045,200 cells.
    for i in range(n_radius-1):
        # loop through the radii: 
        # inner line has radius = radii[i]
        # outer line has radius = radii[i+1]
        if i % 2 ==0:            
            indices_inner_points = np.arange(int(i/2) * n_angle , int(i/2+1) * n_angle , 1)             
            indices_outer_points = np.arange(int(i/2) * n_angle , int(i/2+1) * n_angle , 1) + index_alt
        else:
            indices_inner_points = np.arange(int((i-1)/2) * n_angle , int((i-1)/2 + 1) * n_angle , 1) + index_alt
            indices_outer_points = np.arange(int(i/2+1) * n_angle , int(i/2+2) * n_angle , 1) 
        # Inside Facing       Outside Facing
        # |+---+|              |-------| 
        # |\   /|              |   .   |
        # | \ / |              |  / \  |
        # |  .  |              | /   \ |
        # |-----|              |+-----+|
                 
        outside_facing_triangles = np.array([ indices_inner_points, np.roll(indices_inner_points,(-1)**(i+1)), indices_outer_points ])
        inside_facing_triangles = np.array([  indices_outer_points, np.roll(indices_outer_points,(-1)**(i)),   indices_inner_points ])

        cells = np.hstack([cells, outside_facing_triangles, inside_facing_triangles])

    ufl_mesh    = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))) # shape yields the geometric dimension
    domain      = create_mesh(comm, cells.T, pts.T, ufl_mesh)

    return domain

def bd_outside(x):
    return np.isclose(x[0]**2 + x[1]**2, 4)

def bd_inside(x):
    return np.isclose(x[0]**2 + x[1]**2, 1, atol=.1)

def boundary_2d(x: np.ndarray) -> np.ndarray:
    return np.logical_or(bd_inside(x), bd_outside(x))

if __name__ == "__main__":
    import time
    from sim.common.mesh import circumcenters, is_weakly_acute
    from sim.common.grad_dg0 import check_dg0_grad_central_flux
    import argparse

    parser = argparse.ArgumentParser(description="CLI for the creation of a mesh for a disk with a centered hole.")    
    parser.add_argument('-n_angle', type=int, default = 1, help='Mesh resolution along circumference.')
    parser.add_argument('-n_radius', type=int, default = 1, help='Mesh resolution along radial direction.')    
    parser.add_argument('-nosave', action='store_const', default=False, const=True, help = "Do not safe to XDMF option.")

    args = parser.parse_args()
    
    start   = time.time()
    domain  = create_spiral_mesh(MPI.COMM_WORLD, args.n_angle, args.n_radius)
    end     = time.time()

    print("Mesh creation took", end-start, "seconds.")
    print("Created", domain.geometry.x.shape[0], "nodes and", domain.topology.index_map(domain.topology.dim).size_local, "cells.")

    # CREATE MESHTAGS 

    facets_left   = locate_entities_boundary(domain, 1, bd_inside)
    facets_right  = locate_entities_boundary(domain, 1, bd_outside)   
    
    all_marker = [ np.full(facets_left.size,   2, dtype=np.int32), np.full(facets_right.size,  3, dtype=np.int32) ]
    all_facets= [facets_left, facets_right]

    # Create Mapping:
    facet_indices = np.concatenate(all_facets).astype(np.int32)
    marker_array  = np.concatenate(all_marker)
    # Create Meshtags:
    boundarytags = meshtags(domain, 1, facet_indices, marker_array)

    if args.n_angle == 1:
        filename = f"input/meshes/spiral2D_{args.n_radius}.xdmf"
    else:
        filename = f"input/meshes/spiral2D_{args.n_radius}_{args.n_angle}.xdmf"

    if not (args.nosave):
        with XDMFFile(MPI.COMM_WORLD, filename, "w") as file:
            file.write_mesh(domain)
            file.write_meshtags(boundarytags, domain.geometry)
        print("File saved under ", filename)

    
    cells, centers = circumcenters(domain)
    print("Central Flux is fulfilled:", check_dg0_grad_central_flux(MPI.COMM_WORLD, domain, cells, centers))
    print("Mesh is weakly acute:", is_weakly_acute(domain))