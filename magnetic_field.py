import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Rotate in the x axis
def x_rot(angle):
    rotation_matrix=np.array([[1,0,0],[0, np.cos(angle),-np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    return rotation_matrix

def magnetic_field_calculation(amps):
    # Outer circle radius (distance from eletromagnetic rings to origin)
    outer_circle_radius = 0.35

    # Circle
    Ndeg = 90      # numer of increments for phi
    a = .05          # circle radius
    df = 360.0/Ndeg # differential load of
    dLmag = (df* np.pi / 180) * a

    Ncirc = 6                # number of electromagnetic rings
    omega = 2 * np.pi / Ncirc # differential load of tubulation circle that will dictate the distance bewtween each electromagnetic field ring in degrees

    ########## Analised Points ##########
    Point = [0,0,0] # Reference Point, start (origin)

    nx, ny, nz = (1, 40, 40)          # number of points (resolution)
    xv = np.linspace(0, a, nx) # vector of the points relative to the x axis
    yv = np.linspace(-1*outer_circle_radius, outer_circle_radius, ny) # vector of the points relative to the y axis
    zv = np.linspace(-1*outer_circle_radius, outer_circle_radius, nz) # vector of the points relative to the z axis

    x, y, z = np.meshgrid(xv, yv, zv)

    # Set the dimensions of the cylinder
    cylinder_radius = 0.25
    cylinder_height = 1

    # Calculate the distance from the origin for each point in 3D
    distance_to_axis = np.sqrt(y**2 + z**2)

    # Create a boolean mask for valid points within the cylinder
    valid_points = (distance_to_axis <= cylinder_radius) & (np.abs(x) <= cylinder_height / 2)

    # Create new matrices containing only the valid points
    x_valid = x[valid_points]
    y_valid = y[valid_points]
    z_valid = z[valid_points]

    amp_turns = 200
    I = amps * amp_turns
    print (I)

    ########## Empty Matrices of magnetic force vectors ##########
    H = np.zeros((len(x_valid), 3))
    Hz = np.zeros((len(x_valid)))

    S = np.zeros((Ncirc, Ndeg, 3))

    right_hand = [0,0,1] # right hand orientation at the first ring

    for k in range(len(x_valid)):
        Point[0] = x_valid[k]
        Point[1] = y_valid[k]
        Point[2] = z_valid[k]

        for j in range(Ncirc): # number of circles
            ring_position_d = j * omega
            rot_matrix = x_rot(ring_position_d)

            dH = np.zeros((Ndeg, 3))
            dHmag = np.zeros(Ndeg)
            for i in range(Ndeg):
                f = df*i*np.pi/180 # transform Ndeg from deg to rad
                # point of circle in x*y plane
                xL = a * np.cos(f)
                yL = a * np.sin(f)
                zL = outer_circle_radius

                S[j,i] = (xL, yL, zL)

                # rotation of these points in relation to origin
                S[j,i] = np.dot(S[j,i], rot_matrix)
                rh_orientation = np.dot(right_hand, rot_matrix)

                # distance from circle point to point in space
                Rsuv = S[j,i] / np.linalg.norm(S[j, i])    # unitary vector of origin to circle(point)
                dLuv = np.cross(rh_orientation, Rsuv)      # unitary vector of current
                Puv = np.linalg.norm(Point)                # norm of origin to space (point)
                dL = dLmag*dLuv                            # norm of L
                dHuv = np.cross(rh_orientation, dL)        # unitary vector of dH
                R = [Point - S[j,i]]                       # vector of distance from circle to space (point)
                Rmag = np.linalg.norm(R)                   # norm of R
                Ruv = R/Rmag                               # unitary vector of R
                dH[i,:] = I[j]*np.cross(dL,Ruv)/(4*np.pi*Rmag**2)
                dHmag[i] = np.linalg.norm(dH[i])

            H[k , 0] = np.sum(dH[:,0]) + H[k, 0]
            H[k, 1] = np.sum(dH[:,1]) + H[k, 1]
            H[k, 2] = np.sum(dH[:,2]) + H[k, 2]
            
    # Calculate the magnitude of the magnetic field vectors
    H_mag = np.sqrt(np.sum(H**2, axis=1))

    # Calculate the mean of the magnetic field magnitudes
    med = np.mean(H_mag)

    # Calculate the standard deviation of the magnetic field magnitudes
    Dp = np.std(H_mag)
    
    A = Dp, med
    print(A)

    return Dp, med
    