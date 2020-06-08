################################
#### Solar System Animation ####
################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Math_Project as mpj
#######################
#### Generate Data ####
#######################

h = 300
simulation = mpj.n_body_simulation("five_planets_2d.txt", h = h)
sim_time = 688*24*3600
v = simulation.simulation(sim_time)
v = v / (10**9)
num_planets = simulation.num_planets
shape = (num_planets*2 , len(v[0,:]))
data = np.zeros(shape)
for i in range(num_planets):
    data[i,:] = v[2*i,:]
    data[i+num_planets,:] = v[2*i+1,:]

########################
#### Prepare Figure ####
########################    

fig, ax = plt.subplots(figsize=(5, 4))
bound = int(np.max(data[:,0]) * 1.15)
ax.set(xlim=(-bound, bound), ylim=(-bound, bound))
ax.grid()
c = np.arange(num_planets)
scat = ax.scatter(data[:num_planets,0], data[num_planets:, 0], c = c, cmap = 'rainbow')
sizes = np.ones(num_planets) *10
scat.set_sizes(sizes)

#####################################
#### Scatter Plot (2d) Animation ####
#####################################

jump = 40
def animate(i):
    global data, jump
    scat.set_offsets(np.c_[data[:num_planets,(i*jump)],data[num_planets:,(i*jump)]])    


num_frames = int(sim_time/(h*jump)) - 10
anim = animation.FuncAnimation(fig, animate, interval = 6, frames = num_frames)
plt.draw()
plt.show()
anim.save('My_anim_2d.mp4')

#############
#### END ####
#############




