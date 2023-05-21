import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

#reading the waveform and setting up the time vector
Ts = 0.0000002000 #sampling time
waveform_1 = np.array(pd.read_csv("line0_1_1_8492289.csv", skiprows = 11)) 
waveform_1 = np.insert(waveform_1, 0, np.zeros(101))

waveform_2 = np.array(pd.read_csv("line0_2_1_8492296.csv", skiprows = 11))
waveform_2 = np.insert(waveform_2, 0, np.zeros(136))

waveform_3 = np.array(pd.read_csv("line0_3_1_8492275.csv", skiprows = 11)) 
waveform_3 = np.insert(waveform_3, 0, np.zeros(29))

waveform_4 = np.array(pd.read_csv("line0_4_1_8492269.csv", skiprows = 11))

data = [waveform_1, waveform_2, waveform_3, waveform_4]

#sensor coordinates on a cartesian grid
grid_dim = 15
crd = [[0,0],[grid_dim,0],[grid_dim,grid_dim],[0,grid_dim]]

#AIC function calculator >> returns hit onset time
def calc_AIC(waveform):
    np.seterr(divide='ignore', invalid='ignore') #supresses warnings for edge cases (safe)
    AIC = np.zeros(len(waveform))
    k = 0.00000001 #softening coefficient to avoid numerical instability 
    for i in range(len(waveform)-1):
        i += 1
        AIC[i] = i * np.log10(np.var(waveform[:i]) + k) + (len(waveform) - i - 1) * np.log10(np.var(waveform[i:]) + k)
    
    AIC[0] = AIC[1]
    return np.argmin(AIC) * Ts

onset_arr = np.zeros(4) #hit onset time for each of the four sensors
for i in range(len(onset_arr)):
    onset_arr[i] = calc_AIC(data[i])

def obj_func(position):
    v = 300000 #wave velocity m/s
    err_func = 0
    for i in range(len(onset_arr)):
        delta_obs = onset_arr[i] - onset_arr[0] 
        delta_calc = (np.sqrt((crd[i][0] - position[0])**2 + (crd[i][1] - position[1])**2) - np.sqrt((crd[0][0] - position[0])**2 + (crd[0][1] - position[1])**2)) / v
        err_func += (delta_obs - delta_calc)**2
    
    return err_func

#Particle Swarm Optimization
class particle:
  def __init__(self,dim):
    self.position = np.random.uniform(0.0, grid_dim, dim)
    self.velocity = np.zeros(dim)
    self.error = obj_func(self.position) #current output of objective function
    self.local_best_position = self.position
    self.local_best_error = self.error

#Algorithm Parameters
dim = 2          #Decision variables
max_epochs = 120  #maximum number of iterations
population = 60  #number of particles in swarm
#Hyper-parameters
w = 1  #inertia weight
c1 = 0.1  #cognitive weight
c2 = 0.1  #social weight
VMAX = 0.2*(grid_dim)
VMIN = -VMAX
all_poses = np.zeros((max_epochs,population,dim))
def optimizer():
  global all_poses
  swarm = [particle(dim) for p in range(population)] #creating swarm
  
  global_best_error = np.inf #initializing global best error to "infinity"
  global_best_position = np.zeros(dim)
  #updating global best error
  for i in range(population):
    if swarm[i].error < global_best_error:
      global_best_error = swarm[i].error
      global_best_position = swarm[i].position

  for e in range(max_epochs):
    for j in range(population):
      #updating velocity components for current particle

        #Draw random numbers from a uniform distribution [0,1]
      r1 = np.random.uniform()
      r2 = np.random.uniform()

      swarm[j].velocity = ( w * swarm[j].velocity) + \
      (c1 * r1 * (swarm[j].local_best_position - swarm[j].position)) +  \
      (c2 * r2 * (global_best_position - swarm[j].position))  
      #compute new position using new velocity
      for vel in swarm[j].velocity:
        if vel < VMIN :
          vel = VMIN
        elif vel >= VMAX:
          vel = VMAX

      swarm[j].position += swarm[j].velocity

      for pos in swarm[j].position:
        if pos > 1.5*grid_dim : 
          pos= 1.5*grid_dim
        elif pos < - grid_dim :

          pos = - grid_dim


      all_poses[e,j] = swarm[j].position
      #updating local best position and error
      if (swarm[j].error < swarm[j].local_best_error):
        swarm[j].local_best_error = swarm[j].error
        swarm[j].local_best_position = swarm[j].position
      #updating global best position and error
      if (swarm[j].error < global_best_error):
        global_best_error = swarm[j].error
        global_best_position = swarm[j].position
    
  return global_best_position , global_best_error

sol  = optimizer()
hit_location = sol[0]
print('Function argmin',hit_location) 

#plotting objective function
fig_1 = plt.figure(1)
ax = plt.axes(projection ='3d')
ax.set_title("Objective Function")  
x = y = np.arange(-grid_dim,2*grid_dim,0.1)
X, Y = np.meshgrid(x, y)
Z = obj_func([X,Y])
ax.plot_surface(X, Y, Z)
ax.scatter3D(hit_location[0],hit_location[1] , sol[1])
#plotting contours of the objective function with an estimate of the hit's location
fig_2 = plt.figure(2)
ax2 = plt.axes()

plt.title(label="Objective Function's Contours with the Hit's Location Estimate")
ax2.plot(hit_location[0],hit_location[1],'ro')
ax2.set_xlim(- grid_dim,2*grid_dim)
ax2.set_ylim(- grid_dim,2*grid_dim)
ax2.set_aspect(1)
images =[]
for i in range(len(all_poses)):
    x,y = zip(*all_poses[i])
    image = ax2.scatter(x,y, marker='o' , color='red')
    images.append([image])
animated_image = animation.ArtistAnimation(fig_2, images)
ax2.contour(X,Y,Z,40)
animated_image.save('./test2.gif', writer='pillow') 
plt.show()