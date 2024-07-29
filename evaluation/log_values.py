import neptune

class Logger:
    def __init__(self, algorithm, environment, alpha, gamma):
        self.run = neptune.init_run(
            project="istvan-knab/rl-algorithms",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZGUzMWI0ZC01ZDIyLTQwNWQtODQzOS1mNzQ5NTA3YzdmOGUifQ==",
        )  # your credentials

        self.algorithm = algorithm
        self.environment = environment
        params = {"learning_rate": alpha, "discount_factor": gamma, "Algorithm": algorithm, "Environment": environment}
        self.run["parameters"] = params
        self.run["algorithm"] = algorithm
        self.run["environment"] = environment


    def start_training(self, config):
        print("Starting training------------------")
        print(f"Algorithm: {self.algorithm}")
        print(f"Environment: {self.environment}")
        print(f'Device : {config["DEVICE"]}')
        print(f"Learning Rate: {config['ALPHA']}")
        print(f"Discount Factor: {config['GAMMA']}")
        print(f'Buffer size: {config["BUFFER"]}')
        print(f'Batch size: {config["BATCH"]}')

    def console_log(self, reward, epsilon, episode):
        print('------------------------')
        print('------------------------')
        print("Episode:       ", episode)
        print("Episode reward:", reward)
        print("Epsilon:       ", epsilon)

    def neptune_log(self, reward, epsilon, loss):
        self.run["train/reward"].append(reward)
        self.run["train/epsilon"].append(epsilon)
        self.run["train/loss"].append(loss)


    def step(self, reward, epsilon, episode, loss):
        #self.console_log(reward, epsilon, episode)
        self.neptune_log(reward, epsilon, loss)
        self.run["train/algorithm"] = self.algorithm
        self.run["train/environment"] = self.environment

    def set_tqdm(self):
        WHITE = '\033[97m'
        RESET = '\033[0m'
        tqdm_format = f'{WHITE}{{l_bar}}{{bar}}{{r_bar}}{RESET}'
        return tqdm_format
