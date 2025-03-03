from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Vec2
import numpy as np
import random as rd

# Stałe fizyczne
AU = 149.6e6 * 1000  # 1 AU w metrach
G = 6.67428e-11  # Stała grawitacji
SCALE = 1000 / AU  # Skala: 1 AU = 1000 jednostek w silniku
TIMESTEP = 360  # 1 dzień w sekundach

class MyApp(ShowBase):
    def __init__(self, planets):
        ShowBase.__init__(self)
        self.planets = planets
        self.createPlanets(planets)

        self.mouseSensitivity = 0.1
        self.disableMouse()
        self.cameraDummy = self.render.attachNewNode("cameraDummy")
        self.camera.reparentTo(self.cameraDummy)
        self.camera.setPos(0, 0, 0)
        self.cameraDummy.setPos(0, -50, 0)

        self.taskMgr.add(self.updateSimulation, "updateSimulationTask")

    def createPlanets(self, planets):
        for i, planet in enumerate(planets):
            planet_node = self.render.attachNewNode(f"PlanetNode{i}")
            planet.load_model(self.loader, self.render)
            planet.node = planet_node
            planet.model.setScale(3)  
            planet.model.setPos(*(planet.xyz))  # Przeskalowanie pozycji

    def updateSimulation(self, task):
        for planet in self.planets:
            planet.update_position(self.planets)
            if planet.model:
                planet.model.setPos(*(planet.xyz))  # Aktualizacja pozycji
        return Task.cont

class Planet:
    def __init__(self, position, mass, sun=False, vel_x=0, vel_y=0, vel_z=0, model_path=None): 
        self.mass = mass
        self.xyz = np.array(position, dtype='float64')
        self.vel_xyz = np.array([vel_x, vel_y, vel_z], dtype='float64')

        self.model_path = model_path
        self.model = None
        self.node = None
        self.sun = sun
        self.orbit = []

    def load_model(self, loader, parent):
        self.model = loader.loadModel(self.model_path)
        self.model.reparentTo(parent)
        self.model.setScale(0.5)  # Dopasowanie wielkości planet
        self.model.setPos(*(self.xyz))

    def attracction(self, other):
        diff = other.xyz - self.xyz
        distance = np.linalg.norm(diff)
        if distance == 0:
            return np.array([0, 0, 0], dtype="float64")

        unit_vec = diff / distance
        force = G * self.mass * other.mass / (distance ** 2)
        force_vec = unit_vec * force
        return force_vec

    def update_position(self, planets):
        total_force = np.array([0, 0, 0], dtype="float64")
        for planet in planets:
            if self == planet:
                continue
            
            force_xyz = self.attracction(planet)
            total_force += force_xyz

        self.vel_xyz += total_force / self.mass * TIMESTEP
        self.xyz += self.vel_xyz * TIMESTEP
        self.orbit.append((self.xyz))

if __name__ == "__main__":
    x = y = z = rd.randint(-2,2)
    Earth = Planet([-2,3,2], mass=1000, vel_z=+0.0001, model_path="Models/Earth.gltf")
    Mars = Planet([-3,3,-2], mass=1000, vel_y=-0.0001, model_path="Models/Mars.gltf")
    Jupiter = Planet([5,0,0], mass=20000, vel_x=0.0001, model_path="Models/Jupiter.gltf")

    planets = [Earth, Mars, Jupiter]
    app = MyApp(planets)
    app.run()
