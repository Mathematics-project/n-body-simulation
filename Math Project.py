import numpy as np
import matplotlib.pyplot as plt
from math import pi
import time
class Body():
    """ Holds data on a body (loc_x,loc_y, v_x, v_y, m) in the plane """

    def __init__(self, loc_x=0, loc_y=0, v_x=0, v_y=0, m=0, name=0):
        assert isinstance(loc_x,(int, float)) and isinstance(loc_y,(int, float)) \
        and isinstance(v_x,(int, float)) and isinstance(v_y,(int, float)) \
        and isinstance(m,(int, float)) 
        self.loc = np.array([loc_x, loc_y])
        self.v = np.array([v_x, v_y])
        self.mass = m
        self.name = name

    def __repr__(self):
        return "Body name: " + str(self.name) + '\n' \
            + "location= " + str(self.loc) +'\n' \
            + "Velocity= " + str(self.v) + '\n' 

##########################################################################################################
##########################################################################################################

class n_body_simulation():
    
    def __init__(self, file_name ,h = 1, lst = None , num_planets = 0, G = 6.67 * 10**(-11)):
        self.txt = file_name
        self.planet_lst = lst
        self.h = h
        self.num_planets = num_planets
        self.G = G
    
    def fill_planet(self, line, num):
        seperated_line = line.split()
        self.planet_lst[num] = Body(float(seperated_line[0]), float(seperated_line[1]) ,\
                                    float(seperated_line[2]), float(seperated_line[3]) ,\
                                    float(seperated_line[4]), seperated_line[5])
        return 0
    
    def get_data(self):
        inFile = open(self.txt, "r")
        ## outFile = open("output.txt", "w")
        self.num_planets = int(inFile.readline()[0])
        self.planet_lst = [0]*self.num_planets
        for i in range(self.num_planets):
            line = inFile.readline()
            self.fill_planet(line, i)
        inFile.close()
    
    def r_ij(self, i,j):
        return  np.linalg.norm(self.planet_lst[i].loc - self.planet_lst[j].loc) 

    def direction_i_to_j(self,i,j):
        return self.planet_lst[j].loc - self.planet_lst[i].loc

    def calc_elem_j_in_sum_i(self, i,j):
        return (self.planet_lst[j].mass * self.direction_i_to_j(i, j) )/(self.r_ij(i,j)**3)

    def calc_a_i(self,i):
        res = np.array([0.0,0.0])
        for j in range(self.num_planets):
            if (j!=i):
                res += self.calc_elem_j_in_sum_i(i, j)
        return res * self.G

    def a_lst(self):
        lst = [0]*self.num_planets
        for i in range(self.num_planets):
            lst[i] = self.calc_a_i(i)
        return lst
    
    def euler_step(self):
        a_lst =  self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].loc += self.h*self.planet_lst[i].v
            self.planet_lst[i].v += self.h*a_lst[i]
        
    def leapfrog_pre_first_step(self):
        a_lst = self.a_lst()
        #f1 = a_lst[0] * self.planet_lst[0].mass
        #f2 = a_lst[1] * self.planet_lst[1].mass
        for i in range(self.num_planets):
            self.planet_lst[i].v += 0.5*self.h*a_lst[i]
            
    def leapfrog_step(self):
        for i in range(self.num_planets):
            self.planet_lst[i].loc += self.h*self.planet_lst[i].v
        a_lst = self.a_lst()
        for i in range(self.num_planets):
            self.planet_lst[i].v += self.h*a_lst[i]
        
    def simulation(self, sim_time, alg = "leapfrog"):
        from math import floor
        self.get_data()
        num_of_iter = floor(sim_time / self.h)
        if (alg == "leapfrog"):
            print("leapfrog simulation:\n")
            self.leapfrog_pre_first_step()
            for i in range(num_of_iter):
                self.leapfrog_step()
        else:
            print("euler simulation:\n")
            for i in range(num_of_iter):
                self.euler_step()
        for i in range(self.num_planets):    
            print(self.planet_lst[i].name, "simulated location is: " , self.planet_lst[i].loc)
        print('\n')
        
    def two_body_reference(self, num_1 , num_2, t1):
        from math import sqrt
        from scipy import integrate
        self.get_data()
        m_1 = self.planet_lst[num_1].mass
        m_2 = self.planet_lst[num_2].mass
        x1_0 = self.planet_lst[num_1].loc
        x2_0 = self.planet_lst[num_2].loc
        v1_0 = self.planet_lst[num_1].v
        v2_0 = self.planet_lst[num_2].v
        M = m_1 + m_2
        x_cm = (1/M)*((m_1*x1_0)+(m_2*x2_0))
        v_cm = (1/M)*((m_1*v1_0)+(m_2*v2_0))
        u1_0 = v1_0 - v_cm
        w = x1_0 - x_cm
        r_0 = np.linalg.norm(w)
        alpha = np.arccos((np.inner(u1_0,w))/(np.linalg.norm(u1_0)*np.linalg.norm(w)))
        K = (self.G * (m_2**3))/(M**2)
        u1_0 = np.linalg.norm(u1_0)
        L = r_0 * u1_0 * np.sin(alpha)
        cosgamma = ((L**2)/(K*r_0)) - 1
        singamma = -((L*u1_0*np.cos(alpha))/K)
        gamma = np.arctan2(singamma, cosgamma)
        e = sqrt( ((r_0*(u1_0**2)*(np.sin(alpha)**2)/K)-1)**2 + \
                 (((r_0**2)*(u1_0**4)*(np.sin(alpha)**2)*(np.cos(alpha)**2))/(K**2)) )        
        r_theta = lambda theta: (L**2)/(K*(1+(e*np.cos(theta - gamma))))
        f = lambda x: r_theta(x)**2/L
        #time_lapse = integrate.quadrature(f, 0,2*pi)[0] / (24*3600)
        curr_theta = 0
        max_it = 30
        it = 1
        converged = 0
        while ((it < max_it) & (converged != 1)):
            t_val = integrate.quadrature(f, 0,curr_theta)[0] - t1
            dt_val = f(curr_theta)
            delta = t_val/dt_val
            if (abs(delta)<1.0e-16):
                converged = 1
            curr_theta -= delta
            it += 1 
        x1_new = np.array([r_theta(curr_theta)*np.cos(curr_theta), r_theta(curr_theta)*np.sin(curr_theta)])  
        
        x2_new = -(m_1/m_2)*(x1_new) +(x_cm+v_cm*t1)
        print(self.planet_lst[num_1].name, "reference location is: " , x1_new)
        print(self.planet_lst[num_2].name, "reference location is: " , x2_new)
        print('\n')
        theta = np.linspace(0,2*pi,100)
        res_vec = np.zeros(100)
        x_vec = np.zeros(100)
        y_vec = np.zeros(100)
        for i in range(100):
            res_vec[i] = r_theta(theta[i])
            x_vec[i] = res_vec[i]*np.cos(theta[i])
            y_vec[i] = res_vec[i]*np.sin(theta[i])
        plt.scatter(x_vec, y_vec,c='red')
         
##########################################################################################################
##########################################################################################################

check_two = n_body_simulation("two_body.txt")
solar_system = n_body_simulation("planets.txt")
tt = time.time()
check_two.simulation(19000)
tt = time.time() - tt
print("Simulation time: ",tt)
#check_two.two_body_reference(0,1,1000)
#solar_system.simulation(1000)
#solar_system.simulation(1000, alg = "euler")

