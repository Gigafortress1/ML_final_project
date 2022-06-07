import numpy as np 
import pandas

class point():
    def __init__(self) -> None:
        pass

class road():
    def __init__(self, direction, length, speed, poi=None) -> None:
        self.direction = direction
        self.speed = speed
        self.length = length

        # a dictionary, store the type of poi(key) and their position(value) 
        self.poi = poi
    
    def spend_time(self):
        return self.length / self.speed

class Map():
    def __init__(self, road, start, dest):
        self.road = road
        self.start = start
        self.dest = dest

        # the present position
        self.position = start

    def reset(self):
        return self.start

    def build_map(self):
        pass

    def step(self,action):
        pass

    def render(self):
        self.position = self.start

    