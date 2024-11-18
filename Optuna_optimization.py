import optuna
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import csv

#Rotate in the x axis
def x_rot(angle):
    rotation_matrix=np.array([[1,0,0],[0, np.cos(angle),-np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    return rotation_matrix

def magnetic_field_calculation(amps):
    # Outer circle radius (distance from eletromagnetic rings to origin)
    outer_circle_radius = 0.05

    # Circle
    Ndeg = 45                               # numer of increments for phi
    a = 0.0125                              # circle radius
    df = 360.0/Ndeg                         # differential load of
    dLmag = (df* np.pi / 180) * a

    Ncirc = 2                               # number of electromagnetic rings
    omega = 2 * np.pi / Ncirc               # differential load of tubulation circle that will dictate the distance bewtween each electromagnetic field ring in degrees

    ########## Analised Points ##########
    Point = [0,0,0] # Reference Point, start (origin)

    nx, ny, nz = (1, 20, 20)                                            # number of points (resolution)
    xv = np.linspace(0, a, nx)                                          # vector of the points relative to the x axis
    yv = np.linspace(-1*outer_circle_radius, outer_circle_radius, ny)   # vector of the points relative to the y axis
    zv = np.linspace(-1*outer_circle_radius, outer_circle_radius, nz)   # vector of the points relative to the z axis

    x, y, z = np.meshgrid(xv, yv, zv)

    # Set the dimensions of the cylinder
    cylinder_radius = 0.007
    cylinder_height = 1

    # Calculate the distance from the origin for each point in 3D
    distance_to_axis = np.sqrt(y**2 + z**2)

    # Create a boolean mask for valid points within the cylinder
    valid_points = (distance_to_axis <= cylinder_radius) & (np.abs(x) <= cylinder_height / 2)

    # Create new matrices containing only the valid points
    x_valid = x[valid_points]
    y_valid = y[valid_points]
    z_valid = z[valid_points]

    amp_turns = 1000
    I = amps * amp_turns

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
            
    # Normalize each vector in H to create unit vectors
    H_magnitudes = np.linalg.norm(H, axis=1).reshape(-1, 1)  # Magnitudes of each H vector
    H_unit = H / H_magnitudes  # H as unit vectors

    # Standard vector to compare angles
    standard_vector = np.array([0, 1, 0])

    # Calculate the angle between each H unit vector and the standard vector
    dot_products = np.dot(H_unit, standard_vector)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Angle in radians

    # Return the standard deviation of the angles
    angle_std_dev = np.std(angles)

    # Calculate the mean and standard deviation of the magnetic field magnitudes (not unit vectors)
    H_mag = np.sqrt(np.sum(H**2, axis=1))
    mean_mag = np.mean(H_mag)
    std = np.std(H_mag)

    return angle_std_dev, mean_mag, std
    
nvar = 6
test_inputt=np.zeros((2,6))

def objective(trial):

    var1 = trial.suggest_float("var1", -20.0, 20.0)
    var2 = trial.suggest_float("var2", -20.0, 20.0)
    var3 = trial.suggest_float("var3", -20.0, 20.0)
    var4 = trial.suggest_float("var4", -20.0, 20.0)
    var5 = trial.suggest_float("var5", -20.0, 20.0)
    var6 = trial.suggest_float("var6", -20.0, 20.0)      

    test_inputt = np.array([var1, var2, var3, var4, var5, var6])

    predicted_y_testt = magnetic_field_calculation(test_inputt)
    
    obj1=predicted_y_testt[0]
    obj2=predicted_y_testt[1]
    obj3=predicted_y_testt[2]

    return obj1, obj2, obj3

def convert_to_float(lst):
    result = [float(x) for x in list(lst)]
    return result

# Create study and run it
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(directions=["minimize", "maximize", "minimize"], sampler=sampler)
study.optimize(objective, n_trials=1e4)

print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

i=0
varT=np.zeros((len(study.best_trials),nvar))
for trial in study.best_trials:
    varT[i,:]=convert_to_float(trial.params.values())
    i=i+1

i=0   
objT=np.zeros((len(study.best_trials),3))
for trial in study.best_trials:
    objT[i,:]=(trial.values[0], trial.values[1], trial.values[2])
    i=i+1
    
aux=np.concatenate((varT,objT), axis=1)

# Create .csv
with open('2magnetsn.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ')
    my_writer.writerow(aux)

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Pareto front in 3D
ax.scatter(aux[:, nvar], aux[:, nvar + 1], aux[:, nvar + 2], c='blue', marker='o')

# Set axis labels
ax.set_xlabel('Standard Deviation (Dp)', fontsize=14)
ax.set_ylabel('Mean Intensity (med)', fontsize=14)
ax.set_zlabel('Angle to Standard Vector (radians)', fontsize=14)

# Set ticks for readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Show plot
plt.show()
