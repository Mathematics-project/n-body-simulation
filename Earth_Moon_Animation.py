###################################
#### Solar System 3D Animation ####
###################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Math_Project as mpj
import mpl_toolkits.mplot3d.axes3d as p3

def generate_data(file_name, sim_time, h = 300):
    simulation = mpj.n_body_simulation(file_name, h = h, dim = 3)
    raw_data = simulation.simulation(sim_time)
    num_planets = simulation.num_planets
    jump = 40
    data = [0]*(int(sim_time/(h*jump))+1)
    for i in range(int(sim_time/(h*jump))+1):
        data[i] = np.array(raw_data[0:3,i*jump])
        if (np.linalg.norm((raw_data[3:6,i*jump]*2) - raw_data[0:3,i*jump]) > np.linalg.norm((raw_data[3:6,i*jump]/2)-raw_data[0:3,i*jump])):
            data[i] = np.append([data[i]],[np.array(raw_data[3:6,i*jump])*2.0], axis = 0)
        else:
            data[i] = np.append([data[i]],[np.array(raw_data[3:6,i*jump])/2.0], axis = 0)
    return data, num_planets

def animate_scatters(iteration, data, scatters):
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters

def Earth_Moon_3D_Animation(data,num_planets, out_file, save=False):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    scatters = [ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0])]
    iterations = len(data)
    bound = np.max(np.absolute(data[0])) * 1.15
    ax.set_xlim3d([-bound, bound])
    ax.set_xlabel('X')
    ax.set_ylim3d([-bound, bound])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-bound, bound])
    ax.set_zlabel('Z')
    ax.set_title('3D Animated Solar System')
    # ax.view_init(5, 25)
    num_planets-=1
    sizes = np.ones(num_planets) *6
    for k in range(num_planets):
        scatters[k].set_sizes(sizes)
    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval = 6, blit = False, repeat = True)
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(out_file, writer=writer)
    plt.show()

sim_time = 90*24*3600

#### Generate Data ####
data, num_planets = generate_data("sun_earth_moon.txt", sim_time)

### Animate Data ####
Earth_Moon_3D_Animation(data,num_planets,'tmp_3d.mp4', save = True)

