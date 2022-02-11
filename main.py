import torch
import torchvision.transforms as T
from agent import Agent
from PIL import Image
import numpy as np
import os

import wrapped_flappy_bird as game

game_state = game.GameState()
agent = Agent(gamma=0.99, epsilon=0.05, lr=0.0001, n_actions=2, 
                mem_size=100, batch_size=10, 
                eps_min=0.001, eps_dec=0.000015, replace=50)

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(80),
                    T.ToTensor()])

def img_tensor(img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = resize(img)
        img = torch.Tensor.narrow(img, 2, 0, 112)  
        return img.unsqueeze(0)

image, reward, terminal = game_state.frame_step([1,0])


if os.path.isfile("model_9300_epoch.pt"):
    agent.q_eval.load_state_dict(torch.load(("model_9300_epoch.pt")))
    agent.q_eval.eval()
    print("pretrained model is loaded")

steps = 0
reward_sum = 0
for i in range(10000):
        image, reward, done = game_state.frame_step([1,0])
        last = img_tensor(image)
        current = img_tensor(image)
        state = current - last
        epoche = i
        epsilon = agent.epsilon
        print(f'Epoche:{epoche}, Steps:{steps}, Reward:{reward_sum} Epsilon:{epsilon}')
        steps = 0
        reward_sum = 0
        while not done:
                action, action_pure = agent.action(state)
                _, reward, done = game_state.frame_step(action)
                reward_sum += reward
                for i in range(1):
                        if done == False:
                                image, reward, done = game_state.frame_step(action)
                                reward_sum += reward
                        else: break
                last = current
                current = img_tensor(image)
                next_state = current - last
                agent.save(state, action_pure, reward, next_state, done)
                state = next_state
                agent.learn()
                steps +=1

        if i % 100 == 0: torch.save(agent.q_eval.state_dict(), "model.pt")
        if reward_sum >= 500:
                torch.save(agent.q_eval.state_dict(), "model_final.pt")
                break
