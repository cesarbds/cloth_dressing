from GMM import class_GMM as GMM
import numpy as np
import gym

class SAC_GMM_GMR_Agent():
    def __init__(self, model):
        self.model = model      # Initial model provided
        super(SAC_GMM_GMR_Agent, self).__init__()
    
    def predict_action(self, current_state):
            """
            Predicts action using the integrated GMR method.
            """
            return self.model.predict_action(current_state)

    def get_action_space(self):
        priors_high = np.ones(self.model.priors_size)
        mu_high = np.ones(self.model.mu_size)
        action_high = np.concatenate((priors_high, mu_high), axis=-1)
        action_low = - action_high
        self.action_space = gym.spaces.Box(action_low, action_high)
        #print(self.action_space)
        return action_high, action_low
    
    def update_gaussians(self, gmm_change):
        # change of priors range: [-0.1, 0.1]
        mu_a = self.model.mu[:, :]
        sigma = self.model.sigma[:, :, :]
        priors = gmm_change[:self.model.priors_size]
        priors = priors.reshape(self.model.priors.shape) * 0.5
        print(len(priors))
        # change of mus range: [-0.01, 0.01]
        mu = gmm_change[self.model.priors_size:]
        mu = np.vstack([mu] * 21) * 0.05
        change_dict = {"mu":mu, "priors":priors}
        self.model.update_gaussians(change_dict)
    