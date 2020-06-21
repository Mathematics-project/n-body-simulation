import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp

class Planet():
    """ Holds data on a planet (loc_x,loc_y, v_x, v_y, inclination, m) in the plane """

    def __init__(self, loc_x = 0.0, loc_y = 0.0, v_x = 0.0, v_y = 0.0, inclination = 0, m = 0.0, name = ''):
        """ initialize a body with it's location, velocity, inclination, mass and name """
         
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
    """ after initializing the system, use simulation function to calculate the system's new
    location after t time. Two algorithms are available: Euler and Leapfrog. """ 
    
    def __init__(self, file_name ,h = 1, dim = 2):
        """ expecting a file name. h is the length of a step in both 
        algoritms and the dimention 2D or 3D to simulate """
        cdef double G
        G =  6.67 * pow(10,-11)
        self.file_name = file_name
        self.planet_lst = None
        self.h = h
        self.dim = dim
        self.num_planets = 0
        self.G = G
    
    def create_planet(self, line, num):
        """ creating each planet with it's data in file. Expecting a line of data and num planet in series """
        seperated_line = line.split()
        self.planet_lst[num] = Planet(np.float64(seperated_line[0]), np.float64(seperated_line[1]) ,\
             np.float64(seperated_line[2]), np.float64(seperated_line[3]), np.float64(seperated_line[4]), \
             np.float64(seperated_line[5]), seperated_line[6])
        return 0
    
    def load_data(self):
        """ loads the data received in planet_lst, expecting a file in the following structure:
        num_planets
        location_x  location_y  velocity_x  velocity_y  inclination  mass name
        location_x  location_y  velocity_x  velocity_y  inclination  mass name
        ...  
        ...                                              """ 
        inFile = open(self.file_name, "r")
        self.num_planets = int(inFile.readline())
        self.planet_lst = [0]*self.num_planets
        for i in range(self.num_planets):
            line = inFile.readline()
            self.create_planet(line, i)
        inFile.close()
    
    def turn_data_3D(self):
        """ change 2D data to 3D data using the inclination of each planet 
        with respect to Earth-Sun plane in the Rodrigues fornulae """
        self.dim = 3
        self.load_data()
        for i in range(self.num_planets):
            self.planet_lst[i].loc = np.array([[self.planet_lst[i].loc[0]],[self.planet_lst[i].loc[1]],[0.0]])
            self.planet_lst[i].v = np.array([[self.planet_lst[i].v[0]],[self.planet_lst[i].v[1]],[0.0]])
        K = np.matrix([[0,0,1], [0,0,0], [-1,0,0]])
        I = np.matrix([[1,0,0], [0,1,0], [0,0,1]])
        for i in range(self.num_planets):
            R = I + (np.sin(self.planet_lst[i].inclination*np.pi/360))*K
            + (1-np.cos(self.planet_lst[i].inclination*np.pi/360))*K**2
            self.planet_lst[i].loc = np.squeeze(np.asarray(R*self.planet_lst[i].loc))
            self.planet_lst[i].v = np.squeeze(np.asarray(R*self.planet_lst[i].v))
        
    def two_body_reference(self, t1, num_1 = 0, num_2 = 1):
        """ reference notations are same as Yoel's notes
        using a precision of 64 digits for max accuracy """
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
    
    def a_lst(self):
        """ returns an acceleration list at a specific time. each element
        in res_lst is for the corresponding plantet. """
        cdef int i, j, dim
        dim = self.dim
        res_lst = np.zeros((self.num_planets, dim))
        if dim == 2 :
            """ the case of 2D """
            for i in range(self.num_planets):
                a_i = np.zeros(dim)
                for j in range(self.num_planets):
                    if (j!=i):
                        direction_v = self.planet_lst[j].loc - self.planet_lst[i].loc
                        a_i += (self.planet_lst[j].mass * direction_v)   \
                            /((direction_v[0]**2 + direction_v[1]**2)**1.5)
                res_lst[i] = a_i
            return res_lst * self.G
        else:
            """ the case of 3D """
            for i in range(self.num_planets):
                a_i = np.zeros(3)
                for j in range(self.num_planets):
                    if (j!=i):
                        direction_v = self.planet_lst[j].loc - self.planet_lst[i].loc
                        a_i += (self.planet_lst[j].mass * direction_v)   \
                            /((direction_v[0]**2 + direction_v[1]**2 + direction_v[2]**2)**1.5)
                res_lst[i] = a_i
            return res_lst * self.G
    
    def Euler_step(self):
        """ updates the system location and velocity after one Euler step """
        a_lst =  self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].loc += self.h*self.planet_lst[i].v
            self.planet_lst[i].v += self.h*a_lst[i]
        
    def leapfrog_pre_first_step(self):
        """ before leapfrog first step, the system has to calculate the velocity
        in time 0.5*h without changing planets' location """
        a_lst = self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].v += 0.5*self.h*a_lst[i]
           
    def leapfrog_step(self):
        """ updates the system location and velocity after a Leapfrog step.
        New location in time t, velocity in time t-0.5*h """
        for i in range(self.num_planets):
            self.planet_lst[i].loc += self.h*self.planet_lst[i].v
        a_lst = self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].v += self.h*a_lst[i]
            
    def simulation(self, sim_time, alg = "leapfrog"):
        """ simulates over sim_time with one of the algorithms.
        updates location and velocity at each step.
        Before the simulation starts the system is initialized by given
        data file. 
        The output matrix consists the location for each planet after each step.
        Each planet has 2 or 3 rows (depends on the dimension) for each axis"""
        cdef int i,k,it
        self.load_data()
        sim_dim = self.dim
        if sim_dim == 3 :
            self.turn_data_3D()
        num_of_iter = round(sim_time / self.h)
        """ prepare a data matrix to return """
        shape = (self.num_planets * sim_dim, num_of_iter + 1)
        data_mat = np.zeros(shape)
        for i in range(self.num_planets):
            data_mat[sim_dim*i:sim_dim*(i+1),0] = self.planet_lst[i].loc
        if (alg == "leapfrog"):
            self.leapfrog_pre_first_step()
            for it in range(1,(num_of_iter+1)):
                """ Leapfrog step """
                for k in range(self.num_planets):
                    self.planet_lst[k].loc += self.h*self.planet_lst[k].v
                a_lst = self.a_lst()
                for k in range(self.num_planets):
                    self.planet_lst[k].v += self.h*a_lst[k]
                """ save new locations """
                for i in range(self.num_planets):
                    data_mat[sim_dim*i:sim_dim*(i+1),it] = self.planet_lst[i].loc
        else:
            for it in range(1, (num_of_iter+1)):
                """ Euler step """
                a_lst =  self.a_lst()
                for k in range(self.num_planets):
                    self.planet_lst[k].loc += self.h*self.planet_lst[k].v
                    self.planet_lst[k].v += self.h*a_lst[k]
                """ save new locations """
                for i in range(self.num_planets):
                    data_mat[sim_dim*i:sim_dim*(i+1),it] = self.planet_lst[i].loc
        return data_mat
    
