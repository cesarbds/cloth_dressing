import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import joblib
from scipy.special import comb 

# Definir uma classe que herda de GaussianMixture e adiciona trajetórias suavizadas
class GaussianMixtureWithTrajectories(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params='kmeans', weights_init=None,
                 means_init=None, precisions_init=None, random_state=None,
                 warm_start=False, verbose=0, verbose_interval=10):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol,
                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
                         init_params=init_params, weights_init=weights_init,
                         means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start=warm_start,
                         verbose=verbose, verbose_interval=verbose_interval)
        self.smoothed_trajectory = None

# Função para gerar uma curva de Bézier
def bezier_curve(points, num_points=100):
    """Gera uma curva de Bézier usando os pontos fornecidos."""
    n = len(points) - 1
    t = np.linspace(0, 1, num_points).astype(np.float64)
    curve = np.zeros((num_points, points.shape[1]), dtype=np.float64)

    for i in range(n + 1):
        binomial_coeff = comb(n, i)  # Coeficiente binomial
        curve += binomial_coeff * (t[:, None]**i) * ((1 - t)[:, None]**(n - i)) * points[i]
    
    return curve

# # Função para plotar as trajetórias
# def plot_trajectories_together(original_data, smoothed_data, demo_index):
#     """Plota as trajetórias originais e suavizadas juntas para comparação."""
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))

#     # Plot Trajetória Original
#     ax.plot(original_data[:, 0], original_data[:, 1], label="Original x", color='r', linewidth=2)
#     ax.plot(original_data[:, 0], original_data[:, 2], label="Original y", color='g', linewidth=2)
#     ax.plot(original_data[:, 0], original_data[:, 3], label="Original z", color='b', linewidth=2)

#     # Plot Trajetória Suavizada
#     ax.plot(smoothed_data[:, 0], smoothed_data[:, 1], label="Suavizado x", color='r', linestyle='--', alpha=0.6)
#     ax.plot(smoothed_data[:, 0], smoothed_data[:, 2], label="Suavizado y", color='g', linestyle='--', alpha=0.6)
#     ax.plot(smoothed_data[:, 0], smoothed_data[:, 3], label="Suavizado z", color='b', linestyle='--', alpha=0.6)

#     ax.set_title(f'Trajetória Original e Suavizada - Demonstração {demo_index + 1}')
#     ax.set_xlabel('Tempo (normalizado)')
#     ax.set_ylabel('Posição')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
def main():
    # **Pré-processamento dos dados**
    file_names = ['demo1.csv', 'demo2.csv', 'demo3.csv', 'demo4.csv', 'demo5.csv', 'demo6.csv', 'demo7.csv']
    variables = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    processed_data_list = []
    # Processar cada demonstração individualmente
    for file_name in file_names:
        # 1. Carregar o arquivo CSV
        data = pd.read_csv(file_name)
        
        # 2. Interpolar valores ausentes
        data = data.interpolate(method='linear', axis=0)
        
        # 3. Selecionar apenas as colunas relevantes
        data = data[variables]
        
        # 4. Normalizar os dados (Min-Max Scaling entre 0 e 1)
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        
        # 5. Adicionar uma dimensão temporal normalizada
        time = np.linspace(0, 1, len(normalized_data)).reshape(-1, 1)
        demo_data = np.hstack((time, normalized_data))
        
        # 6. Adicionar a demonstração processada à lista
        processed_data_list.append(demo_data)

    print(f"Processamento concluído. Total de demonstrações: {len(processed_data_list)}")

    # **Treinamento do GMM com seleção de componentes usando BIC**
    bic_scores = []
    optimal_components = []

    for i, demo_data in enumerate(processed_data_list):
        print(f"Avaliando número de componentes para a demonstração {i + 1}")
        n_components_range = range(1, 21)
        demo_bic_scores = []
        best_bic = np.inf
        best_n_components = None
        
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=50)
            gmm.fit(demo_data)
            bic = gmm.bic(demo_data)
            demo_bic_scores.append(bic)
            if bic < best_bic:
                best_bic = bic
                best_n_components = n

        optimal_components.append(best_n_components)
        bic_scores.append(demo_bic_scores)

    # Para salvar:
    joblib.dump(scaler, 'scaler.pkl')

    # Para carregar:
    scaler = joblib.load('scaler.pkl')


    # **Treinamento do GMM usando os componentes otimizados**
    gmm_models = []
    for i, (demo_data, n_components) in enumerate(zip(processed_data_list, optimal_components)):
        print(f"Treinando GMM com {n_components} componentes para a demonstração {i + 1}")
        gmm = GaussianMixtureWithTrajectories(n_components=n_components, covariance_type='full', reg_covar=1e-3, random_state=50)
        gmm.fit(demo_data)
        gmm_models.append(gmm)

    # Lista para armazenar as trajetórias suavizadas
    smoothed_trajectories = []

    # Gerar amostras suavizadas do GMM e plotar
    for i, gmm in enumerate(gmm_models):
        print(f"Gerando amostras para demonstração {i + 1} com GMM")

        # 1. Gerar amostras do GMM
        num_samples = 150
        samples, _ = gmm.sample(num_samples)
        samples = samples[np.argsort(samples[:, 0])]  # Ordenar pelo tempo

        # 2. Suavizar a trajetória usando Bézier
        smoothed_trajectory = []
        for col_idx in range(1, samples.shape[1]):  # Ignorar o tempo na primeira coluna
            points = np.column_stack((samples[:, 0], samples[:, col_idx]))  # Pontos de controle (tempo e variável)
            curve = bezier_curve(points, num_points=num_samples)
            smoothed_trajectory.append(curve[:, 1])  # Adicionar a variável suavizada

        # Recombinar tempo e variáveis suavizadas
        smoothed_trajectory = np.column_stack([samples[:, 0]] + smoothed_trajectory)
        gmm.smoothed_trajectory = smoothed_trajectory  # Adicionar a trajetória suavizada ao modelo
        smoothed_trajectories.append(smoothed_trajectory)

        # Carregar os dados originais da demonstração
        demo_data = processed_data_list[i]

        # # Plotar as trajetórias originais e suavizadas juntas
        # plot_trajectories_together(demo_data, smoothed_trajectory, i)

    # Opcional: Salvar as trajetórias suavizadas
    np.save('smoothed_trajectories.npy', smoothed_trajectories)
    print("Trajetórias suavizadas salvas em 'smoothed_trajectories.npy'")
    print(np.array(smoothed_trajectories).shape)

    # Salvar os modelos GMM com as trajetórias suavizadas
    joblib.dump(gmm_models, 'gmm_models_with_smoothed_trajectories.pkl')
    print("Modelos GMM com trajetórias suavizadas salvos com sucesso em 'gmm_models_with_smoothed_trajectories.pkl'")


if __name__ == "__main__":
    main()