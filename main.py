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
        f = 1.0 - r_s / self.r

        # O E é a energia por unidade de massa (afinidade, neste caso)
        dt_dlambda = sqrt((self.dr**2) / (f**2) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dlambda

        # Trilha de pontos
        self.rastro = [(self.x, self.y)]

    def __repr__(self):
        return f"Raio (r = {self.r:.2e}. phi = {self.phi:.2f}, dr = {self.dr:.2e}, dphi = {self.dphi:.2e})"
    
    def passo(self, d_lambda, rs):
        """ Avança o raio em um passo d_lambda usando RK4 """
        if self.r <= rs:
            return # Encerra se estiver dentro do horizonte de eventos
        
        # Integração RK4
        self.r, self.phi, self.dr, self.dphi = rk4_passo(self, d_lambda, rs)
        
        # Converter de volta para cartesiano
        self.x = self.r * cos(self.phi)
        self.y = self.r * sin(self.phi)

        # Registrar a trilha 
        self.rastro.append((self.x, self.y))

# --- Funções de Integração RK4 --- #
def rhs_geodesico(raio, rs):
    """
    Calcula o Lado Diteito (Right Hand Side - RHS) das EDOs para as geodésicas nulas
    Estados: y = [r, phi, dr/dλ, dphi/dλ]
    RHS: [dr/dλ, dphi/dλ, d²r/dλ², d²φ/dλ²]
    """

    r = raio.r
    dr = raio.dr
    dphi = raio.dphi
    E = raio.E

    f = 1.0 - rs / r

    # 1. dr/dλ
    rhs0 = dr

    # 2. dφ/dλ
    rhs1 = dphi

    # d²r/dλ² (equação de movimento radial)
    dt_dlambda = E / f
    rhs2 = (
        - (rs / (2 * r**2)) * f * (dt_dlambda**2)
        + (rs / (2 * r**2 * f)) * (dr**2)
        + (r - rs) * (dphi**2)
    )

    # d²φ/dλ² (equação de movimento angular)
    rhs3 = -2.0 * dr * dphi / r

    return np.array([rhs0, rhs1, rhs2, rhs3])

def rk4_passo(raio, d_lambda, rs):
    """
    Executa um passo de integração Runge-Kutta de 4ª Ordem.
    Retorna o novo estado [r, phi, dr, dphi]
    """

    y0 = np.array([raio.r, raio.phi, raio.dr, raio.dphi])

    # K1
    k1 = rhs_geodesico(raio, rs)

    # K2    
    # Cria uma cópia dentro do raio com o estado intermediário y0 + k1 * (d_lambda / 2)
    temp2 = y0 + k1 * (d_lambda / 2.0)

    # C++: Ray r2 = ray; r2.r=temp[0]; ...
    r2 = Raio((raio.x, raio.y), (0, 0), rs) # Cópia base
    r2.r, r2.phi, r2.dr, r2.dphi = temp2
    r2.E = raio.E # E é uma constante de conservação
    k2 = rhs_geodesico(r2, rs)

    # K3
    temp3 = y0 + k2 * (d_lambda / 2.0)
    r3 = Raio((raio.x, raio.y), (0, 0), rs) 
    r3.r, r3.phi, r3.dr, r3.dphi = temp3
    r3.E = raio.E
    k3 = rhs_geodesico(r3, rs)

    # K4
    temp4 = y0 + k3 * d_lambda
    r4 = Raio((raio.x, raio.y), (0, 0), rs)
    r4.r, r4.phi, r4.dr, r4.dphi = temp4
    r4.E = raio.E
    k4 = rhs_geodesico(r4, rs)

    # Novo Estado
    ynovo = y0 + (d_lambda / 6.0) * (k1 + 2 * k2 * 2 + k3 + k4)

    # Retorna o novo estado (r, phi, dr, dphi)
    return ynovo[0], ynovo[1], ynovo[2], ynovo[3]

def main_simulacao():
    """
    Função principal que configura o sistema e executa um passo de simulação
    """

    # Buraco Negro (Sagittarius A* - Sgr A*)
    # Massa: 4.3 milhões de massas solares ≈ 8.54e36 kg
    SagA = BuracoNegro((0.0, 0.0), 8.54e36)

    print(SagA)
    print(f"Raio de Schwarzschild (r_s): {SagA.r_s:.2e} metros")

    # Parâmetros de Simulação
    D_LAMBDA = 1.0e2 # Passo de afinidade (a afinidade é a 'variável de tempo' para raios de luz)
    NUM_PASSOS = 50000 # Número de passos de integração

    

    # Definindo os parâmetros iniciais fixos
    inicio_x = -1.0e11 # Posição X inicial
    # Raio inicial
    # Exemplo: um raio a uma distância (x, y) = (-1e11, 3.27e10)
    # E viajando com uma velocidade c em (1,0)
    posicao_inicial = (-1e11, 3.27606302719999999e10)
    direcao_inicial = (c, 0.0) # Velocidade (dx/dt, dy/dt)

    # Definindo as posições Y para os raios 
    min_y = -2.0e10
    max_y = 4.0e10
    num_raios = 30

    # Lista que armazena todos os raios
    raios = []

    for i in range(num_raios):
        # Calcula a posição Y de forma linear 
        inicio_y = min_y + (max_y - min_y) * (i / (num_raios - 1))

        posicao_inicial = (inicio_x, inicio_y)

        # Cria e adiciona o novo raio à lista
        novo_raio = Raio(posicao_inicial, direcao_inicial, SagA.r_s)
        raios.append(novo_raio)
    
    print(f"Iniciando a simulação com {len(raios)} raios...")

    # Lista de índices dos raios que AINDA estão ativos (fora do BN)
    raios_ativos_indices = list(range(len(raios)))

    for i in range(NUM_PASSOS):
        # Quebra o loop se todos os raios pararam
        if not raios_ativos_indices:
            print(f"Todos os araios atingiram o horizonte ou saíram no passo {i+1}")
            break
        raios_a_remover = []
        
        for idx in raios_ativos_indices:
            raio = raios[idx]

            # Executa o passo de integração (AGORA SERÁ EXECUTADO!)
            raio.passo(D_LAMBDA, SagA.r_s)

            # Checa se o raio caiu no horizonte de eventos
            if raio.r <= SagA.r_s:
                raios_a_remover.append(idx)
        
        for idx in sorted(raios_a_remover, reverse=True):
            raios_ativos_indices.remove(idx)
    
    print("Simulação concluída")

    # --- Visualização com Matplotlib --- #
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Simulação de Raio de Luz em Buraco Negro de Schwarzschild ({NUM_PASSOS} passos)")
    ax.set_xlabel("Distância X (m)")
    ax.set_ylabel("Distância Y (m)")
    ax.set_aspect('equal', adjustable='box')

    # Desenha o Buraco Negro (Horizonte de Eventos)
    bn_circulo = plt.Circle(SagA.posicao, SagA.r_s, color='black', fill=True, label='Horizonte de Eventos')
    ax.add_artist(bn_circulo)

    # Ajustar limites do gráfico para cobrir o buraco negro e o rastro 
    todos_x = [SagA.posicao[0]]
    todos_y = [SagA.posicao[1]]

    # Itera sobre TODOS os raios da lista 'raios'
    for raio in raios:
        if raio.rastro: # Usando 'rastro' conforme seu código
            rastro_x, rastro_y = zip(*raio.rastro)

            # Adiciona os pontos para o ajuste dos limites do gráfico 
            todos_x.extend(rastro_x)
            todos_y.extend(rastro_y)

            ax.plot(rastro_x, rastro_y, color='magenta', linewidth=2, alpha=0.8)

    min_x, max_x = min(todos_x), max(todos_x)
    min_y, max_y = min(todos_y), max(todos_y)

    # Adicionar uma margem
    margem = 0.1 * max(max_x - min_x, max_y - min_y)
    ax.set_xlim(min_x - margem, max_x + margem)
    ax.set_ylim(min_y - margem, max_y + margem)

    plt.show()

# --- Execução ---#
if __name__ == "__main__":
    main_simulacao()