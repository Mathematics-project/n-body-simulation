import numpy as np
import matplotlib.pyplot as plt
import time
from mpmath import mp

class Planet():
    """ Holds data on a body (loc_x,loc_y, v_x, v_y, m) in the plane """

    def __init__(self, loc_x = 0, loc_y = 0, v_x = 0, v_y = 0, inclination = 0, m = 0, name = ''):
        # initialize a body with it's location, velocity, mass and name
         
        self.loc = np.array([loc_x, loc_y])
        self.v = np.array([v_x, v_y])
        self.inclination = inclination
        self.mass = m
        self.name = name

    def __repr__(self):
        return "Planet name: " + str(self.name) + '\n' \
            + "location= " + str(self.loc) +'\n' \
            + "Velocity= " + str(self.v) + '\n' 

##########################################################################################################
##########################################################################################################

class n_body_simulation():
    # after initializing the system, use simulation function to calculate the system's new
    # location after t time. Two choices are available: Euler and Leapfrog algorithms. 
    
    def __init__(self, file_name ,h = 1, dim = 2):
        # expecting a file name. h argument is the length of a step in both algoritms.
        self.file_name = file_name
        self.planet_lst = None
        self.h = h
        self.dim = dim
        self.num_planets = 0
        self.G = np.float64(6.67 * 10**(-11))
    
    def fill_planet(self, line, num):
        # creating each planet with it's location,velocity,mass and name
        seperated_line = line.split()
        self.planet_lst[num] = Planet(np.float64(seperated_line[0]), np.float64(seperated_line[1]) ,\
             np.float64(seperated_line[2]), np.float64(seperated_line[3]), np.float64(seperated_line[4]), \
             np.float64(seperated_line[5]), seperated_line[6])
        return 0
    
    def load_data(self):
        # fills the data received in planet_lst, expecting a file in the following structure:
        # num_planets
        # loc1_x  loc1_y  v_x  v_y  mass name
        # following line are the same structure for next planets
        inFile = open(self.file_name, "r")
        self.num_planets = int(inFile.readline()[0])
        self.planet_lst = [0]*self.num_planets
        for i in range(self.num_planets):
            line = inFile.readline()
            self.fill_planet(line, i)
        inFile.close()
    
    def turn_data_3d(self):
        self.dim = 3
        self.load_data()
        for i in range(self.num_planets):
            self.planet_lst[i].loc = np.array([[self.planet_lst[i].loc[0]],[self.planet_lst[i].loc[1]],[0.0]])
            self.planet_lst[i].v = np.array([[self.planet_lst[i].v[0]],[self.planet_lst[i].v[1]],[0.0]])
        K = np.matrix([[0,0,1], [0,0,0], [-1,0,0]])
        I = np.matrix([[1,0,0], [0,1,0], [0,0,1]])
        for i in range(self.num_planets):
            R = I + (np.sin(self.planet_lst[i].inclination))*K + (1-np.cos(self.planet_lst[i].inclination))*K**2
            self.planet_lst[i].loc = np.squeeze(np.asarray(R*self.planet_lst[i].loc))
            self.planet_lst[i].v = np.squeeze(np.asarray(R*self.planet_lst[i].v))
        
    def two_body_reference(self, t1, num_1 = 0, num_2 = 1):
        # reference notations are same as Yoel's notes
        from scipy import integrate
        mp.dps = 64
        self.load_data()
        t1 = mp.mpf(t1)
        m_1 = mp.mpf(self.planet_lst[num_1].mass)
        m_2 = mp.mpf(self.planet_lst[num_2].mass)
        x1_0 = mp.matrix(self.planet_lst[num_1].loc)
        x2_0 = mp.matrix(self.planet_lst[num_2].loc)
        v1_0 = mp.matrix(self.planet_lst[num_1].v)
        v2_0 = mp.matrix(self.planet_lst[num_2].v)
        M = m_1 + m_2
        x_cm = (1/M)*((m_1*x1_0)+(m_2*x2_0))
        v_cm = (1/M)*((m_1*v1_0)+(m_2*v2_0))
        u1_0 = v1_0 - v_cm
        w = x1_0 - x_cm
        r_0 = mp.norm(w)
        alpha = mp.acos((np.inner(u1_0,w))/(mp.norm(u1_0)*mp.norm(w)))
        K = mp.mpf(self.G) * (m_2**3)/(M**2)
        u1_0 = mp.norm(u1_0)
        L = r_0 * u1_0 * mp.sin(alpha)
        cosgamma = ((L**2)/(K*r_0)) - 1
        singamma = -((L*u1_0*mp.cos(alpha))/K)
        gamma = mp.atan2(singamma, cosgamma)
        e = mp.sqrt(((r_0*(u1_0**2)*(mp.sin(alpha)**2)/K)-1)**2 + \
                 (((r_0**2)*(u1_0**4)*(mp.sin(alpha)**2)*(mp.cos(alpha)**2))/(K**2)) )        
        r_theta = lambda theta: (L**2)/(K*(1+(e*mp.cos(theta - gamma))))
        f = lambda x: r_theta(x)**2/L
        curr_theta = 0
        max_it = 30
        it = 1
        converged = 0
        while ((it < max_it) & (converged != 1)):
            t_val = mp.quad(f,[0,curr_theta]) - t1
            # t_val = integrate.quadrature(f, 0,curr_theta)[0] - t1
            dt_val = f(curr_theta)
            delta = t_val/dt_val
            if (abs(delta)<1.0e-6):
                converged = 1
            curr_theta -= delta
            it += 1 
        x1_new = mp.matrix([r_theta(curr_theta)*mp.cos(curr_theta), r_theta(curr_theta)*mp.sin(curr_theta)]) 
        # x2_new = -(m_1/m_2)*(x1_new) +(x_cm + v_cm*t1)
        x1_new = x1_new + (x_cm + v_cm*t1)
        return x1_new
        
    # def r_ij(self, i,j):
    #     # returns the distance between body i and body j
    #     return  np.linalg.norm(self.planet_lst[i].loc - self.planet_lst[j].loc) 

    # def direction_i_to_j(self,i,j):
    #     # returns the direction vector from body i to body j
    #     return self.planet_lst[j].loc - self.planet_lst[i].loc
    
    # def calc_elem_j_in_sum_i(self, i,j):
    #     # calculates the element (m_j * r_ij)/(|r_ij|^3)
    #     return (self.planet_lst[j].mass * self.direction_i_to_j(i, j) )/(self.r_ij(i,j)**3)

    # def calc_a_i(self,i):
    #     # returns a_i (acceleration of body i) by summing all elements and multyplying by G
    #     res = np.zeros(2)
    #     for j in range(self.num_planets):
    #         if (j!=i):
    #             res += self.calc_elem_j_in_sum_i(i, j)
    #     return res * self.G

    # def a_lst(self):
    #     # returns a list of each body's acceleration
    #     lst = [0]*self.num_planets
    #     for i in range(self.num_planets):
    #         lst[i] = self.calc_a_i(i)
    #     return lst
    
    def a_lst(self):
        # returns an acceleration list of each body
        lst = [0]*self.num_planets
        if self.dim ==2 :
            for i in range(self.num_planets):
                a_i = np.zeros(self.dim)
                for j in range(self.num_planets):
                    if (j!=i):
                        dist_v = self.planet_lst[i].loc - self.planet_lst[j].loc
                        a_i += (self.planet_lst[j].mass * (self.planet_lst[j].loc - self.planet_lst[i].loc))   \
                            /((dist_v[0]**2 + dist_v[1]**2)**1.5)
                lst[i] = a_i * self.G
            return lst
        else:
            for i in range(self.num_planets):
                a_i = np.zeros(self.dim)
                for j in range(self.num_planets):
                    if (j!=i):
                        dist_v = self.planet_lst[i].loc - self.planet_lst[j].loc
                        a_i += (self.planet_lst[j].mass * (self.planet_lst[j].loc - self.planet_lst[i].loc))   \
                            /((dist_v[0]**2 + dist_v[1]**2 + dist_v[2]**2)**1.5)
                lst[i] = a_i * self.G
            return lst
    
    def euler_step(self):
        # updates the system location and velocity after an Euler step
        a_lst =  self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].loc += self.h*self.planet_lst[i].v
            self.planet_lst[i].v += self.h*a_lst[i]
        
    def leapfrog_pre_first_step(self):
        # before leapfrog first step, the system calculates the velocity
        # in time 0 + 0.5 without changing location
        a_lst = self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].v += 0.5*self.h*a_lst[i]
           
    def leapfrog_step(self):
        # updates the system location and velocity after an Euler step
        # location in time, velocity in time - 0.5
        res = [0]*self.num_planets
        for i in range(self.num_planets):
            self.planet_lst[i].loc += self.h*self.planet_lst[i].v
            res[i] = self.planet_lst[i].loc 
        a_lst = self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].v += self.h*a_lst[i]
        return res
                
    # def simulation(self, sim_time, alg = "leapfrog"):
    #     # simulates over sim_time with one of the algorithms.
    #     # updates location and velocity at each step
    #     # before the simulation starts the system is initialized by given
    #     # data file. the output matrix consists the location for each planet.
    #     # after each step. Each planet has two rows in the result matrix, one
    #     # for x_loc and one for y_loc
    #     self.load_data()
    #     num_of_iter = round(sim_time / self.h)
    #     shape = (self.num_planets * 2, num_of_iter + 1)
    #     data_mat = np.zeros(shape)
    #     for i in range(self.num_planets):
    #         data_mat[2*i:2*i+2,0] = self.planet_lst[i].loc
    #     if (alg == "leapfrog"):
    #         self.leapfrog_pre_first_step()
    #         for it in range(1,(num_of_iter+1)):
    #             self.leapfrog_step()
    #             for i in range(self.num_planets):
    #                 data_mat[2*i:2*i+2,it] = self.planet_lst[i].loc
    #     else:
    #         for it in range(1, (num_of_iter+1)):
    #             self.euler_step()
    #             for i in range(self.num_planets):
    #                 data_mat[2*i:2*i+2,it] = self.planet_lst[i].loc
    #     return data_mat
    
    def simulation(self, sim_time, alg = "leapfrog"):
        # simulates over sim_time with one of the algorithms.
        # updates location and velocity at each step
        # before the simulation starts the system is initialized by given
        # data file. the output matrix consists the location for each planet.
        # after each step. Each planet has two rows in the result matrix, one
        # for x_loc and one for y_loc
        self.load_data()
        sim_dim = self.dim
        if sim_dim == 3 :
            self.turn_data_3d()
        num_of_iter = round(sim_time / self.h)
        shape = (self.num_planets * sim_dim, num_of_iter + 1)
        data_mat = np.zeros(shape)
        for i in range(self.num_planets):
            data_mat[sim_dim*i:sim_dim*(i+1),0] = self.planet_lst[i].loc
        if (alg == "leapfrog"):
            self.leapfrog_pre_first_step()
            for it in range(1,(num_of_iter+1)):
                for k in range(self.num_planets):
                    self.planet_lst[k].loc += self.h*self.planet_lst[k].v
                a_lst = self.a_lst()
                for k in range(self.num_planets):
                    self.planet_lst[k].v += self.h*a_lst[k]
                for i in range(self.num_planets):
                    data_mat[sim_dim*i:sim_dim*(i+1),it] = self.planet_lst[i].loc
        else:
            for it in range(1, (num_of_iter+1)):
                # a_lst =  self.a_lst()
                # for k in range(self.num_planets):
                #     self.planet_lst[k].loc += self.h*self.planet_lst[k].v
                #     self.planet_lst[k].v += self.h*a_lst[k]
                self.euler_step()
                for i in range(self.num_planets):
                    data_mat[sim_dim*i:sim_dim*(i+1),it] = self.planet_lst[i].loc
        return data_mat
    
#############################################################################
#### check how many simulation steps occcur over one second in real time ####
#############################################################################

# 2-body result: 27,000 simulations per second at h=1 
# 6-body result: 3,400 simulations per second at h=1

def check_sim_time(n, file_name, h = 1, dim = 2):
    sim = n_body_simulation(file_name, h = h, dim = dim)
    t1 = time.time()
    res = sim.simulation(n)
    t2 = time.time()
    print(t2 - t1)
    return res

# data_2 = check_sim_time(27000, file_name = "sun_earth_2d.txt", h = 1, dim = 3)
# data_6 = check_sim_time(3400,file_name = "six_planets_2d.txt", h = 1, dim = 3)
        
#############################################################################
#### Plot Error (simulation vs reference) to delta_time in one algorithm ####
#############################################################################
    
def err_to_del_t(k,jump,normalized = False ,alg = "leapfrog"):
    err = np.zeros(k)
    h = 0.0
    check_two = n_body_simulation("sun_earth_2d.txt")
    for i in range(k):
        h = h + jump
        t = h
        check_two.h = h
        ref_loc_e = np.array(check_two.two_body_reference(t))
        check_two.simulation(t, alg = alg)
        sim_loc_e = check_two.planet_lst[0].loc
        if normalized == False:
            err[i] = np.log(np.linalg.norm(ref_loc_e - sim_loc_e))
        else:
            err[i] = np.linalg.norm(ref_loc_e - sim_loc_e)/np.linalg.norm(ref_loc_e)
    return err

def print_err_del_t(jump, limit = 400, normalized = False):
    k = int( limit / jump)
    err1 = err_to_del_t(k,jump, normalized)
    err2 = err_to_del_t(k,jump, normalized , alg = "euler")
    x_axis = [np.log((1 + i)*jump) for i in range(k)]
    figure_1 = plt.figure()
    figure_1.suptitle('Error to delta time')
    ax = plt.subplot(1,2,1)
    figure_1 = plt.plot(x_axis,err1, color = 'r')
    plt.title('Leapfrog')
    ax = plt.subplot(1,2,2)
    figure_1 = plt.plot(x_axis,err2, color = 'b')
    plt.title('Euler')
    slope1_1, intercept1_1 = np.polyfit(x_axis[40:], err1[40:], 1)
    slope1_2, intercept1_2 = np.polyfit(x_axis, err2, 1)
    return (err1, err2, slope1_1, intercept1_1, slope1_2, intercept1_2)
    
# (err1_1, err1_2, s1, in1, s2, in2) = print_err_del_t(5)
# (err2_1, err2_2) = print_err_del_t(0.0001, normalized = True)

###############################################################
#### Simulation error using leapfrg algorithm through time ####
###############################################################
    
def error_to_time(sim_time, h = 1, alg = "leapfrog", file_name = "sun_earth_2d.txt", relative = True):
    # receives sim_time in second, h, algorithm to use, and a file name.
    # returns the error over time
    
    sim = n_body_simulation(file_name, h = h)
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
    t = 17
    return err

def plot_err_time(time, h_min, h_max, jump = 1,file_name = "sun_earth_2d.txt", alg = 'leapfrog', relative = True):
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

sim_time = 4*365*24*3600
# err_in_time = plot_err_time(time = sim_time, h_min = 600, h_max = 600, jump = 100)
# err_in_time = plot_err_time(time = sim_time, h_min = 1, h_max = 1, jump = 1, file_name="six_planets_2d.txt")

# err_in_time = plot_err_time(time = sim_time, h_min = 1200, h_max = 1200, jump = 100, alg = 'euler')

##########################################################################################################
##########################################################################################################


# from mpl_toolkits.mplot3d import axes3d

# fig2 = plt.figure()
# ax = fig2.add_subplot(111,projection = '3d')
# x = [1,2,3,4,5,6,7,8]
# y = [2,4,6,8,10,12,14,16]
# z = [3,6,9,12,15,18,21,24]

# ax.scatter(x,y,z, c='b', marker='o')

# plt.show()




