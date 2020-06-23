import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
import Math_Project as mpj

""" For using cython, follow instructions in setup.py file 
un-comment the next line to use the Cython compiled version """

# import Math_Project_Cython as mpj

#############################################################################
#### check how many simulation steps occur over one second in real time ####
#############################################################################

def check_sim_time(n, file_name, h = 1, dim = 2):
    """ check the time of a simulation with specific time, file, h and dimension """
    import time
    sim = mpj.n_body_simulation(file_name, h = h, dim = dim)
    t1 = time.time()
    res = sim.simulation(n)
    t2 = time.time()
    print(t2 - t1)
    return res

data_2 = check_sim_time(310000, file_name = "sun_earth.txt", dim = 2)
data_6 = check_sim_time(40000,file_name = "six_planets.txt", dim = 2)
        
#############################################################################
#### Plot Error (simulation vs reference) to delta_time in one algorithm ####
#############################################################################
    
def single_step_err(k, jump, alg = "leapfrog", relative = False):
    """ returns an error vector of k elements. For each h from jump to (k-1)*jump
    we simulate one simulation step and fill err with earth's location error
    with respect to the reference """
    err = np.zeros(k)
    h = 0.0
    sun_earth_sim = mpj.n_body_simulation("sun_earth.txt")
    for i in range(k):
        h = h + jump
        t = h
        sun_earth_sim.h = h
        ref_loc_e = np.array(sun_earth_sim.two_body_reference(t))
        sun_earth_sim.simulation(t, alg = alg)
        sim_loc_e = sun_earth_sim.planet_lst[0].loc
        err[i] = mp.log(np.linalg.norm(ref_loc_e - sim_loc_e))
        if relative:
            err[i] = err[i] - mp.log(np.linalg.norm(ref_loc_e))
    return err

def print_err(jump, limit, relative = False):
    """ Prints two plots, one for each algorithm.
    Every plot is using err_ro_h with k=limit/jump 
    Returns the slope and interception of each of the lines """
    k = int( limit / jump)
    err1 = single_step_err(k, jump,alg = "leapfrog", relative = relative)
    err2 = single_step_err(k, jump, alg = "euler", relative = relative)
    x_axis = [np.log((1 + i)*jump) for i in range(k)]
    figure_1 = plt.figure()
    figure_1.suptitle('Log(Error) / Log(time step)')
    ax = plt.subplot(1,2,1)
    figure_1 = plt.plot(x_axis, err1, color = 'r')
    plt.title('Leapfrog')
    ax = plt.subplot(1,2,2)
    figure_1 = plt.plot(x_axis, err2, color = 'b')
    plt.title('Euler')
    slope1_1, intercept1_1 = np.polyfit(x_axis[30:], err1[30:], 1)
    slope1_2, intercept1_2 = np.polyfit(x_axis, err2, 1)
    return (err1, err2, slope1_1, intercept1_1, slope1_2, intercept1_2)
    
# (err1_1, err1_2, slope_Leapfrog, inp1, slope_Euler, inp2) = print_err(jump = 20, limit = 1200)
# (err2_1, err2_2, slope1, inp1, slope2, inp2) = print_err(jump=20, limit = 1200, relative = True)

###############################################################
#### Simulation error using leapfrg algorithm through time ####
###############################################################
    
def error_to_time(sim_time, h = 1, alg = "leapfrog", file_name = "sun_earth.txt", relative = True):
    """ Calculates error of earth-sun simulation over time """
    sim = mpj.n_body_simulation(file_name, h = h)
    one_day = 3600 * 24
    days = int(sim_time / one_day)
    data = sim.simulation(sim_time, alg = alg)
    err = np.zeros(days + 1)
    for i in range(days + 1):
        t = i*one_day
        t_data = int(t/h)
        ref_loc_e = np.array(sim.two_body_reference(t))
        sim_loc_e = np.array([data[0,t_data], data[1,t_data]])
        tmp_vec = ref_loc_e - sim_loc_e 
        err[i] = (tmp_vec[0]**2 + tmp_vec[1]**2)**0.5
        if (relative == True):
            err[i] = err[i]/((ref_loc_e[0]**2 + ref_loc_e[1]**2)**0.5)
    return err

def plot_err_time_diff_h(time, h_min, h_max, jump = 1,file_name = "sun_earth.txt", alg = 'leapfrog', relative = True):
    """ For h from h_min to h_max with jump size of jump variable, plot the error over sim_time with all h's
    in that range """
    col_list = ['red', 'green', 'blue' ,'purple','yellow', 'black', 'pink', 'grey']
    max_hs = len(col_list)
    assert int(np.ceil((h_max - h_min + jump)/jump)) < max_hs
    one_day = 3600*24
    for h in range(h_min , h_max + 1,jump):
        err = error_to_time(sim_time = time , h = h, alg = alg ,file_name = file_name, relative = relative)   
        x_axis = np.arange((time / one_day ) + 1)
        plt.plot(x_axis, err, color = col_list[int((h - h_min)/jump)], label = h)
    plt.title('Relative error with different h / days')
    plt.legend()
    plt.show()    
    return 0

# sim_time = 1*365*24*3600
# err_in_time = plot_err_time_diff_h(time = sim_time, h_min = 400, h_max = 400, jump = 100)

#########################################################
#### Creating a random planets file with random data ####
#########################################################
    
def create_random_planets_file(num_planets):
    """ make a real-like file with num_planets planets filled with random data """
    outFile = open("outfile.txt", "w")
    outFile.write(str(num_planets)+'\n')
    for i in range(num_planets):
        lst = np.random.rand(6) *np.random.randint(10,100000)
        for k in lst:
            outFile.write(str(k)+ " ")
        outFile.write('aa\n')
    outFile.close()
    
# create_random_planets_file(1000)


