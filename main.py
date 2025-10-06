import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, atan2, pi

# --- Constantes Globais --- #
c = 299792458.0        # Velocidade da luz (m/s)
G = 6.67430e-11        # Constante Gravitacional (N*(m/kg)^2)

# O código original em C++ usa GLM vec3 e vec2. 
# Neste código python irei utilizar arrays/listas/tuplas.
# Para posições (x, y) irei utilizar tuplas de 2 elementos.

# --- Classes --- #
class BuracoNegro:
    """ Representação do Buraco Negro de Schwarzschild """

    def __init__(self, posicao, massa):
        
        # Posição é uma tupla (x, y)
        self.posicao = posicao
        self.massa = massa

        # Raio de Schwarzschild (r_s)
        self.r_s = 2.0 * G * self.massa / (c * c)

        def __repr__ (self):
            return f"Buraco Negro (M = {self.massa:.2e}kg, R_s={self.r_s:.2e}m)"

class Raio:
    """ Representa um raio de luz (geodésica nula) """

    def __init__(self, posicao, direcao, r_s):
        # Posicao e Direcao são tuplas (x, y) para caresiano 
        self.x, self.y = posicao

        # Coordenadas polares iniciais
        self.r = sqrt(self.x**2 + self.y**2)
        self.phi = atan2(self.y, self.x)

        # Velocidades iniciais (componentes dr/dλ e dφ/dλ)
        # O código original usa dir.x e dir.y (velocidade cartesiana) para calcular dr e dphi
        dir_x, dir_y = direcao
        self.dr = dir_x * cos(self.phi) + dir_y * sin(self.phi)
        self.dphi = (-dir_x * sin(self.phi) + dir_y * cos(self.phi)) / self.r

        # Quantidades conservadas
        self.L = self.r**2 * self.dphi # Momento angular
        f = 1.0 - rs / self.r

        # O E é a energia por unidade de massa (afinidade, neste caso)
        dt_dlambda = sqrt((self.dr**2) / (f**2) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dlambda

        # Trilha de pontos
        self.trail = [(self.x, self.y)]

    def __repr__(self):
        return f"Raio (r = {self.r:.2e}. phi = {self.phi:.2f}, dr = {self.dr:.2e}, dphi = {self.dphi:.2e})"
    
    def passo(self, d_lambda, rs):
        """ Avança o raio em um passo d_lambda usando RK4 """

        