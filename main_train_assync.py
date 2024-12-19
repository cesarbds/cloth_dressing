from env.dress_environment_assync import DressEnv
from SAC_cleanrl.sac_cleanrl import *
from TD3_cleanrl.td3_cleanrl import *
from TD3_cleanrl.TD3_agent_thread import TD3_agent
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
import threading
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

###parameters declaration
seed = random.randint(0,1000000)
torch_deterministic = True
buffer_size: int = int(1e5)
cuda = True
print(seed)

# TRY NOT TO MODIFY: seeding
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
import queue
device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
# Fila para comunicação entre o agente e o ambiente
action_queue = queue.Queue()
obs_queue = queue.Queue()
start_queue = queue.Queue()
action_lock = threading.Lock()
obs_lock = threading.Lock()
start_lock = threading.Lock()
reset_queue = queue.Queue()

# Função que executa o ambiente em loop
def run_environment(env):

    while True:
        obs,_,_ = env.reset()  # Inicializa o ambiente
        
        with start_lock:
            print("Reseted")
            start_queue.put(1)
        with obs_lock:
            reset_queue.put(obs)
        done  = False
        action = np.array([0,0,0,0,0,0,0])
        episode_ts = 0
        while not done:
            start = time.time()

            with action_lock:
                try:
                    action, episode_ts = action_queue.get(timeout=.01)  # The agent sends the action
                except queue.Empty:
                    a=1#continue
            
            new_state, reward, done,_ , _ = env.step(action) # Environment responds with new state and reward

            # Acquire lock before putting the result into the observation queue
            with obs_lock:
                obs_queue.put((new_state, reward, done))  # Send the result to t
            #env.render()  # Renderiza o estado
            if done or episode_ts > 2.5e2:
                print("Ambiente finalizado.")
                break




if __name__ == "__main__":

    # Initialize the environment
    env = DressEnv()
    agent = TD3_agent(env=env, enable_logging = True, seed=seed, action_queue = action_queue, obs_queue = obs_queue, action_lock = action_lock, obs_lock = obs_lock, start_queue = start_queue, start_lock = start_lock, reset_queue=reset_queue)
    # Create a thread
    thread = threading.Thread(target=run_environment, args=(env,), daemon=True)
    thread2 = threading.Thread(target=agent.train)
    # Start the thread
    thread.start()
    thread2.start()

    

    #agent.train()
    
    env.close()
