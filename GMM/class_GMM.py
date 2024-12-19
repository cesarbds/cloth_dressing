import os
import joblib
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, model_name=None, model_index=6):
        self.model_index = model_index  # Índice do modelo a ser carregado
        if model_name is not None:
            if not os.path.isfile(model_name):
                raise Exception(f'GMM model file "{model_name}" not found')
            _, file_extension = os.path.splitext(model_name)
            if file_extension == ".pkl":
                self.load_model(model_name)
            else:
                raise Exception("Extension not supported. Use .pkl files.")

    def load_model(self, model_name):
       
        if os.path.isfile(model_name):
            self.model = joblib.load(model_name)  # Isso deve carregar uma lista de modelos GMM
            # Seleciona o modelo baseado no índice fornecido
            self.gmm = self.model[self.model_index]

            # Agora, podemos acessar os parâmetros do modelo selecionado
            self.priors = np.array(self.gmm.weights_)
            #print(f"hmmmmmmmmmmmmmmmmmmmm: {self.model}")

            #print(f"####################################################{self.priors.shape}")
            self.priors_size = self.priors.shape[0]
            self.mu = np.array(self.gmm.means_)
            self.mu_size = self.mu.shape[1] - 1
            self.sigma = np.array(self.gmm.covariances_)
            self.sigma_size = self.sigma.shape[1] - 1
            print(f"Model {self.model_index} loaded successfully")
        else:
            print("File doesn't exist")

    def get_parameters(self):
        gmm_state = {
            'priors': self.priors,
            'mu': self.mu,
            'sigma': self.sigma
        }
        return gmm_state

    def get_main_gaussian(self, x):
        weights = self.get_weights(x)
        k = np.argmax(weights)
        return k, self.priors[k], self.mu[:, k], self.sigma[:, :, k]
    
    def update_gaussians(self, change_dict):
        d_priors = change_dict["priors"]
        self.gmm.weights_ += d_priors
        self.gmm.weights_[self.gmm.weights_ < 0] = 0
        self.gmm.weights_ /= self.gmm.weights_.sum()  # Normalize to sum to 1

        d_mu = change_dict["mu"]
        d_mu = d_mu.reshape(self.gmm.means_.shape)
        self.gmm.means_ += d_mu

    def update_main_gaussian(self, x, d_mu):
        """
        Atualiza a Gaussiana principal com base no maior peso.
        """
        weights = self.get_weights(x)  # Calcula os pesos de todas as Gaussianas
        k = np.argmax(weights)  # Identifica a Gaussiana com maior peso

        # Atualiza a média (mu) da Gaussiana principal k
        d_mu = d_mu[:13]  # Certifique-se de que d_mu tem 13 características
        self.mu[k, :] += d_mu  # Atualizar a média removendo a primeira característica (tempo)



    def get_weights(self, x):
        """
        Calcula os pesos de cada Gaussiana dado o estado x.
        """
        # print(x.shape)
        self.mu = self.gmm.means_
        self.priors = self.gmm.weights_
        self.sigma = self.gmm.covariances_
        num_gaussians = self.mu.shape[0]
        # print(f"state_mu shape before removing time: {self.mu.shape}")
        # print(f"state_sigma shape before removing time: {self.sigma.shape}")
        state_mu = self.mu[:, :]
        state_sigma = self.sigma[:, :, :]
        

        # print(f"state_mu shape after removing time: {state_mu.shape}")
        # print(f"state_sigma shape after removing time: {state_sigma.shape}")

        weights = np.zeros((num_gaussians))
        for i in range(num_gaussians):
            #print(f"X: {x}")
            #print(f"Media predita: {state_mu[i,:]}")
            #print(f"Sigma: {state_sigma[i,:]}")
            weights[i] = self.priors[i] * multivariate_normal.pdf(x, mean=state_mu[i,:], cov=state_sigma[i,:,:])
            # print(f"Prior {i}: {self.priors[i]}")
            # print(f"PDF {i}: {multivariate_normal.pdf(x, mean=state_mu[i], cov=state_sigma[i])} ")
        weights /= (np.sum(weights, axis=0) + np.finfo(float).eps)
        
        return weights



    def predict_trajectory(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]
        dim = x.shape[1]
        num_gaussians = self.mu.shape[2]

        weights = self.get_weights(x)
        trajectory_mean = np.zeros((batch_size, dim))

        for i in range(num_gaussians):
            state_mu = self.mu[:, i, :dim]
            state_sigma = self.sigma[:, i, :dim, :dim]
            trajectory_mean += weights[i].reshape(-1, 1) * state_mu

        return trajectory_mean.squeeze()

    def predict_joint_pos(self, x):
        """ 
        Input
        x: np_array representing the current state relative to the target (state_dim) or (state_dim,)
        Output
        joint_pos_mean: np_array represing the predicted joint_pos (State_dim) or (state_dim,)
        """
        
        dim = x.shape # number of joints
        num_gaussians = self.mu.shape[0]
        mu = self.mu[:, 1:]
        #assert dim - 1 == self.mu.shape[1]
        sigma = self.sigma[:, 1:, 1:]
        weights = self.get_weights(x)
        # print(f"pesos: {weights}")
        
        state_mean = np.zeros((1,dim[0]))
        for i in range(num_gaussians):
            state_mu = mu[i, :]
            #joint_pos_mu = self.mu[i,6:dim]
            state_sigma = sigma[i, :, :]
            #jp_sigma = self.sigma[i, 6:dim, 6:dim]
            aux = state_mu + (state_sigma @ np.linalg.pinv(state_sigma) @ (x - state_mu).T).T # batch_size x dim
            # print(f"Aux: {aux}")
            state_mean += weights[i].reshape(-1, 1) * aux
            # print(f"STATE MEAN: {state_mean}")
        return state_mean.squeeze()
        
    def save_model(self, model_name):
        joblib.dump(self.model, model_name)
        print(f"Model saved as {model_name}")
