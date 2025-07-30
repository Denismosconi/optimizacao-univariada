import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def funcao_objetivo(X, Q):
    """Funcao objetivo"""
    return np.dot(X.T, np.dot(Q, X))

def gerar_indice_arquivo(pasta, prefixo, extensao):
    """Gera um índice incremental para nome de arquivo não sobrescrever."""
    i = 1
    while True:
        nome = f"{prefixo}_{i}.{extensao}"
        caminho = os.path.join(pasta, nome)
        if not os.path.exists(caminho):
            return i
        i += 1

def univariate_optm(Vetor0, Vetorf, Q, passo=0.1, tol=1e-6, max_iter=200):
    historicoX = [Vetor0]
    X = Vetor0 - Vetorf
    funcao = funcao_objetivo(X, Q)
    historicoF = [funcao]

    for i in range(max_iter):
        if i % 2 == 0:
            S = passo * np.array([1, 0])
        else:
            S = passo * np.array([0, 1])

        X_atual = Vetor0 - Vetorf
        X_direita = (Vetor0 + S) - Vetorf
        X_esquerda = (Vetor0 - S) - Vetorf

        f_atual = funcao_objetivo(X_atual, Q)
        f_direita = funcao_objetivo(X_direita, Q)
        f_esquerda = funcao_objetivo(X_esquerda, Q)

        if f_direita < f_atual:
            Vetor0 = Vetor0 + S
        elif f_esquerda < f_atual:
            Vetor0 = Vetor0 - S
        else:
            print(f"Convergência alcançada na iteração {i+1}")
            break

        X = Vetor0 - Vetorf
        funcao = funcao_objetivo(X, Q)
        historicoX.append(Vetor0.copy())
        historicoF.append(funcao)
        if funcao < tol:
            print(f"Convergência alcançada na iteração {i+1}")
            break
    else:
        print("Número máximo de iterações alcançado")

    return Vetor0, funcao, historicoX, historicoF, i + 1

def plot_otimizacao_3d(historicoX, historicoF, Q, Vetorf, save_plot=True):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    historico_absoluto = [p for p in historicoX]

    x_min = min(p[0] for p in historico_absoluto) - 1
    x_max = max(p[0] for p in historico_absoluto) + 1
    y_min = min(p[1] for p in historico_absoluto) - 1
    y_max = max(p[1] for p in historico_absoluto) + 1

    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    def funcao_plot(x, y):
        vec = np.array([x, y]) - Vetorf
        return np.dot(vec.T, np.dot(Q, vec))

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = funcao_plot(X[i, j], Y[i, j])

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    x_hist = [p[0] for p in historico_absoluto]
    y_hist = [p[1] for p in historico_absoluto]
    z_hist = [funcao_objetivo(np.array(p) - Vetorf, Q) for p in historico_absoluto]

    ax.scatter(x_hist, y_hist, z_hist, c='r', s=50, label='Iterações')
    ax.plot(x_hist, y_hist, z_hist, c='r', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.set_title('Trajetória da Otimização em 3D')
    plt.legend()

    if save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        indice = gerar_indice_arquivo('plots', 'otimizacao_3d', 'png')
        plt.savefig(f'plots/otimizacao_3d_{indice}.png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_curvas_nivel(historicoX, Q, Vetorf, save_plot=True):
    plt.figure(figsize=(10, 8))

    historico_absoluto = [p for p in historicoX]

    x_min = min(p[0] for p in historico_absoluto) - 1
    x_max = max(p[0] for p in historico_absoluto) + 1
    y_min = min(p[1] for p in historico_absoluto) - 1
    y_max = max(p[1] for p in historico_absoluto) + 1

    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    def funcao_plot(x, y):
        vec = np.array([x, y]) - Vetorf
        return np.dot(vec.T, np.dot(Q, vec))

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = funcao_plot(X[i, j], Y[i, j])

    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Valor da função objetivo')

    x_hist = [p[0] for p in historico_absoluto]
    y_hist = [p[1] for p in historico_absoluto]
    plt.scatter(x_hist, y_hist, c='r', s=20, label='Iterações')
    plt.plot(x_hist, y_hist, 'r-')
    plt.scatter(Vetorf[0], Vetorf[1], c='b', s=100, marker='*', label='Ponto Final')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Curvas de Nível e Trajetória da Otimização')
    plt.legend()

    if save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        indice = gerar_indice_arquivo('plots', 'curvas_nivel', 'png')
        plt.savefig(f'plots/curvas_nivel_{indice}.png', dpi=300, bbox_inches='tight')

    plt.show()

def salvar_dados(historicoX, historicoF, num_iteracoes):
    if not os.path.exists('dados'):
        os.makedirs('dados')

    indice = gerar_indice_arquivo('dados', 'historicoX', 'txt')

    with open(f'dados/historicoX_{indice}.txt', 'w') as f:
        for ponto in historicoX:
            f.write(f"{ponto[0]}, {ponto[1]}\n")

    with open(f'dados/historicoF_{indice}.txt', 'w') as f:
        for valor in historicoF:
            f.write(f"{valor}\n")

    with open(f'dados/iteracoes_{indice}.txt', 'w') as f:
        f.write(f"{num_iteracoes}")

if __name__ == "__main__":
    while True:
        print("\n=== Otimização para Função de Duas Variáveis ===")
        print("Por favor, informe as coordenadas dos dois pontos:")

        x0 = float(input("Digite o valor de x0: "))
        y0 = float(input("Digite o valor de y0: "))
        xf = float(input("Digite o valor de xf: "))
        yf = float(input("Digite o valor de yf: "))

        Vetor0 = np.array([x0, y0])
        Vetorf = np.array([xf, yf])

        Q = np.identity(2)
        print("\nA matriz Q foi inicializada como matriz identidade:")
        print(Q)

        alterar = input("\nDeseja alterar os valores da matriz Q? (s/n): ").strip().lower()
        if alterar == 's':
            print("\nInforme os novos valores da diagonal da matriz Q (2x2):")
            q11 = float(input("Digite Q[1,1]: "))
            q22 = float(input("Digite Q[2,2]: "))
            Q = np.array([[q11, 0], [0, q22]])

        Vetor_optm, funcao_optm, historicoX, historicoF, num_iter = univariate_optm(Vetor0, Vetorf, Q)

        print(f"\nPonto ótimo encontrado: (x, y) = ({Vetor_optm[0]:.6f}, {Vetor_optm[1]:.6f})")
        print(f"Valor da função no ponto ótimo: {funcao_optm:.6f}")
        print(f"Número de iterações: {num_iter}")
        print("\nHistórico de pontos:")
        for i, (x, y) in enumerate(historicoX):
            print(f"Iteração {i}: ({x:.6f}, {y:.6f})")
        print("\nHistórico de função objetivo:")
        for i, fx in enumerate(historicoF):
            print(f"Iteração {i}: ({fx:.6f})")

        plot_otimizacao_3d(historicoX, historicoF, Q, Vetorf, save_plot=True)
        plot_curvas_nivel(historicoX, Q, Vetorf, save_plot=True)

        salvar_dados(historicoX, historicoF, num_iter)

        repetir = input("\nDeseja executar outra otimização? (s/n): ").strip().lower()
        if repetir != 's':
            break
