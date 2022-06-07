from cmath import sqrt
import numpy as np 
import pandas

class point():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def distance(self, p):
        return sqrt(pow(self.x - p.x, 2)+pow(self.y - p.y,2))

class road():
    def __init__(self, p1, p2, length, speed, poi=None) -> None:
        # the two vertexs of the road
        self.p1 = p1
        self.p2 = p2
        
        self.speed = speed
        self.length = p1.distance(p2)

        # a dictionary, store the type of poi(key) and their position(value) 
        self.poi = poi
    
    def spend_time(self):
        return self.length / self.speed

class Map():
    def __init__(self, road, start):
        # a set of roads
        self.road = road
        # start of the customer
        self.start = start

        # the present position
        self.position = start

    def reset(self):
        return self.start

    def step(self, action):
        # After finishing the road meeting a crossing road, 
        # choose the direction according to the action
        
        pass

    def render(self):
        self.position = self.start

    