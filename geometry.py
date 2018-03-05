import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from copy import deepcopy

def dist(p,q):
    return np.sqrt((q.x-p.x)**2 + (q.y-p.y)**2)

class Point(object):
    def __init__(self, x,y, dist=dist):
        self.x = x
        self.y = y
        self.d = dist
    def angle(self, p):
        try:
            return np.angle((p.x-self.x)+1j*(p.y-self.y))
        except:
            print("error in Point()",p, self)
            return np.angle((p.x-self.x)+1j*(p.y-self.y))
    def distance(self, p):
        return self.d(self,p)
    
    def same_side_of_as(self,l, p):
        r_self = l.a*self.x +l.b*self.y +l.c
        r_p = l.a*p.x +l.b*p.y +l.c
        if r_self*r_p > 0:
            return True
        else:
            return False
    def draw(self, ax, **kwargs):
        ax.scatter([self.x],[self.y], **kwargs)
    def __repr__(self):
        return "("+str(self.x)+", "+str(self.y)+")"


class Line(object):
    def __init__(self, ):
        self.a = self.b = self.c = 1
    def from_param(self,a,b,c):
        if not((a==0 and b==0)):
            self.a = a
            self.b = b
            self.c = c
    def from_points(self, A,B):
        if not (A.x==B.x and A.y==B.y):
            if A.y==B.y:
                self.a=0
                self.c=A.y
                self.b=1
            else:
                self.a = 1
                self.b = -(A.x-B.x)/(A.y-B.y)
                self.c = 0 - (A.x + self.b*A.y)
        else:
            print("Line Error: Same points", A, B)
            return False
    def random_point(self):
        if self.a!=0:
            y = random.random()
            x = -(self.b*y+self.c)/self.a
            return Point(x, y)
        else:
            x = random.random()
            return Point(x, -self.c/self.b)
    def from_pointsbetween(self, A, B):
        x1, y1 = A.x, A.y
        x2, y2 = B.x, B.y
        self.a = -2*(x1-x2)
        self.b = -2*(y1-y2)
        self.c = (x1**2-x2**2)+(y1**2-y2**2)
    def intersection(self, l):
        if (self.a * l.b == self.b * l.a):
            return "parallel"
        else:
            solution = np.linalg.solve(np.matrix([[self.a,self.b],[l.a,l.b]]), -np.matrix([[self.c],[l.c]]))
            r = Point(solution[0,0], solution[1,0])
            return r
      
    def f(self, x):
        if self.b!=0:
            return -(x*self.a+self.c)/self.b
        else:
            return None
    def draw(self, ax, rang, **kwargs):
        ax.plot(rang, [self.f(x) for x in rang], **kwargs)
    #def simmetric_point(self, p):
    
class Poligon(object):
    def __init__(self, p):
        self.p=p
        self.dots = []
        self.lines = []
        self.dot_names = []
        
    def load_dots(self, dots, dot_names):
        for dot, name in zip(dots, dot_names):
            self.dots.append(deepcopy(dot))
            self.dot_names.append(deepcopy(name))
        try:
            self.dots, self.dot_names =  zip(*sorted(zip(self.dots, self.dot_names), 
                                                    key = lambda var: self.p.angle(var[0])))
        except:
            print("error Poligon: self.dots, self.dot names", self.dots, self.dot_names)
        self.dots = list(self.dots)
        self.dot_names = list(self.dot_names)
        
    def produce_lines_from_dots(self): 
        self.lines.clear()
        for dot in self.dots:
            self.lines.append(Line())
            self.lines[-1].from_pointsbetween(self.p, dot)
            
    def kick_redundant_dots_lines(self):
        redundant_list = []
        i = 0
        while i< len(self.dots):
            redundant = True
            parallel_lines = []
            for j in range(len(self.dots)):
                if j!=i:
                    intersection = self.lines[i].intersection(self.lines[j])
                    intersection_good = True
                    if intersection=="parallel":
                        parallel_lines.append(j)
                    else:    
                        k=0
                        while k < len(self.dots) and intersection_good:
                            if k !=i and k!=j:
                                if not intersection.same_side_of_as(self.lines[k], self.p):
                                    intersection_good=False
                            k+=1
                        if intersection_good:
                            redundant=False
            if not redundant:
                for j in parallel_lines:
                    if not self.p.same_side_of_as(self.lines[j],self.lines[i].random_point()):
                        redundant = True
            if redundant:    
                redundant_list.append(self.dot_names[i])
                del self.lines[i]
                del self.dot_names[i] 
                del self.dots[i]
                
            else:
                i+=1
        return redundant_list