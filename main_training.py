from env.dress_environment import DressEnv
from SAC_cleanrl.sac_cleanrl import *
from TD3_cleanrl.td3_cleanrl import *
from TD3_cleanrl.TD3_agent import TD3_agent
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

###parameters declaration
seed = random.randint(0,1000000)
torch_deterministic = True
buffer_size: int = int(1e5)
cuda = True


# TRY NOT TO MODIFY: seeding
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

if __name__ == "__main__":

    # Initialize the environment
    env = DressEnv()
    agent = TD3_agent(env=env, seed=seed)

    agent.train()
    
    env.close()
