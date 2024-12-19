import os
import joblib
import numpy as np
from scipy.stats import multivariate_normal


class GMM_GMR:
    def __init__(self, model_name=None, model_index=0):
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
        self.model = joblib.load(model_name)
        self.gmm = self.model[self.model_index]
        self.priors = np.array(self.gmm.weights_)
        self.mu = np.array(self.gmm.means_)
        self.sigma = np.array(self.gmm.covariances_)
        self.priors_size = self.priors.shape[0]
        self.mu_size = self.mu.shape[1]
        self.sigma_size = self.sigma.shape[1]

    def gmr(self, current_state):
        """
        Gaussian Mixture Regression (GMR) to infer next action based on the current state.
        """
        n_components = len(self.priors)
        dim_input = len(current_state)
        dim_output = self.mu_size - dim_input

        predicted_mean = np.zeros(dim_output)
        predicted_covariance = np.zeros((dim_output, dim_output))

        for i in range(n_components):
            mean_input = self.mu[i, :dim_input]
            mean_output = self.mu[i, dim_input:]
            cov_input = self.sigma[i, :dim_input, :dim_input]
            cov_output = self.sigma[i, dim_input:, dim_input:]
            cov_input_output = self.sigma[i, :dim_input, dim_input:]

            inv_cov_input = np.linalg.pinv(cov_input)

            predicted_mean += self.priors[i] * (
                mean_output + cov_input_output.T @ inv_cov_input @ (current_state - mean_input)
            )
            predicted_covariance += self.priors[i] * (
                cov_output - cov_input_output.T @ inv_cov_input @ cov_input_output
            )

            #print(f"media: {predicted_mean}")
        return predicted_mean, predicted_covariance
    


    def _predict_action(self, current_state):
        """
        Predict the next action using GMR.
        """
        predicted_action, predicted_covariance = self.gmr(current_state)
        
        #predicted_mean, _ = self.gmr(current_state)
        #print(f"predicted_mean: {predicted_mean}")
        return predicted_action
    
    def update_gaussians(self, change_dict):
        d_priors = change_dict["priors"]
        d_mu = change_dict["mu"]

        # Adicionar logs para verificar as mudanças
        #print(f"Antes da atualização - Priors: {self.gmm.weights_}, Mu: {self.gmm.means_}")

        self.gmm.weights_ += d_priors
        self.gmm.weights_ = np.clip(self.gmm.weights_, 0, None)
        self.gmm.weights_ /= self.gmm.weights_.sum()

        self.gmm.means_ += d_mu

        #print(f"Após a atualização - Priors: {self.gmm.weights_}, Mu: {self.gmm.means_}")

