###################################
#### Solar System 2D Animation ####
###################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Math_Project as mpj

def generate_data(file_name, sim_time, h = 300):
    """ generates the data to animate, receives a file_name to load data and simulation time 
    returns data where the first rows are x axis location and last rows are y axis location """
    simulation = mpj.n_body_simulation(file_name, h = h, dim = 2)
    raw_data = simulation.simulation(sim_time)
    num_planets = simulation.num_planets
    shape = (num_planets*2 , len(raw_data[0,:]))
    data = np.zeros(shape)
    for i in range(num_planets):
        data[i,:] = raw_data[2*i,:]
        data[i+num_planets,:] = raw_data[2*i+1,:]
    return data, num_planets

def animate(i, data, scat, num_planets):
    jump = 40
    scat.set_offsets(np.c_[data[:num_planets,(i*jump)],data[num_planets:,(i*jump)]])    

def Solar_System_2D_Animation(data, num_planets, sim_time, h, out_file):
    """ animate data in 2D, receives number of planets to animate, simulation time, h,
    and an out file name to save the animation"""
    fig, ax = plt.subplots(figsize=(5, 4))
    bound = int(np.max(data[:,0]) * 1.15)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound))
    ax.grid()
    c = np.arange(num_planets)
    scat = ax.scatter(data[:num_planets,0], data[num_planets:, 0], c = c, cmap = 'rainbow')
    sizes = np.ones(num_planets) *10
    scat.set_sizes(sizes)
    jump = 40
    num_frames = int(sim_time/(h*jump)) - 10
    anim = animation.FuncAnimation(fig, animate, interval = 6, frames = num_frames, fargs = (data, scat, num_planets))
    plt.draw()
    plt.show()
    anim.save(out_file)


h = 300
sim_time = 2*365*24*3600

#### Generate Data ####
data, num_planets = generate_data("five_planets.txt", sim_time)

#### Animate Data ####
Solar_System_2D_Animation(data,num_planets, sim_time, h,'my_anim_2d.mp4')
    