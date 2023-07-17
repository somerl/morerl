import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import wandb
from wandb.integration.sb3 import WandbCallback

#мы заменили float и np.array
class GoLeftEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left.
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code
  FIRST = 0
  SECOND = 1
  THIRD = 2
  FORTH = 3
  FIFTH= 4
  SIXTH = 5
  SEVENTH = 6

  def __init__(self, grid_size=10, render_mode="console"):
    super(GoLeftEnv, self).__init__()
    self.render_mode = render_mode

    self.fields = np.array([[0, 0, 7, 7,], [1, 0, 0, 0,], [2, 0, 0, 0,], [3, 0, 0, 0,], [4, 1, 0, 0,], [5, 0, 0, 0], [6, 0, 0, 0], [7, 0, 0, 0], [8, 1, 0, 0], [9, 0, 0, 0], [10, 0, 0, 0], [11, 0, 0, 0], [12, 0, 0, 0],
                   [13, 0, 0, 0], [14, 1, 0, 0], [15, 0, 0, 0], [0, 0, 0, 0]])

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    n_actions = 7
    self.action_space = spaces.Discrete(n_actions)
    # The observation will be the coordinate of the agent
    # this can be described both by Discrete and Box space
    self.observation_space = spaces.Box(low=-100, high=100,
                                        shape=(17,4), dtype=np.int8)
  #попробовать переделать всё в np.array

  def reset(self, seed=None, options=None):
    """
    Important: the observation must be a numpy array
    :return: (np.array)
    """
    super().reset(seed=seed, options=options)
    # Initialize the agent at the right of the grid
    self.fields = np.array([[0, 0, 7, 7,], [1, 0, 0, 0,], [2, 0, 0, 0,], [3, 0, 0, 0,], [4, 1, 0, 0,], [5, 0, 0, 0], [6, 0, 0, 0], [7, 0, 0, 0], [8, 1, 0, 0], [9, 0, 0, 0], [10, 0, 0, 0], [11, 0, 0, 0], [12, 0, 0, 0],
                   [13, 0, 0, 0], [14, 1, 0, 0], [15, 0, 0, 0], [0, 0, 0, 0]])

    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return self.fields.astype(np.int8), {}

  def step(self, action):


     def first():
            numberforcycles = 15

            balcycle1 = 0

            roll = self.fields[16][0]

            lst = [0, 1, 2, 3, 4]
            weights = [6.25, 25, 37.5, 25, 6.25]
            nextroll = random.choices(lst, weights=weights, k=1)
            nextroll = nextroll[0]
            self.fields[16][0] = nextroll
            #print("Выпала" , roll)
            arrplayer1 = []
            while balcycle1 < numberforcycles:
              ballnow = self.fields[balcycle1][2]
              ballindex = self.fields[balcycle1][0]
          #расположение фишек игрока 1
              if ballnow != 0:
               arrplayer1.append(ballindex)
              balcycle1 += 1
            if self.fields[8][3] == 1:
                arrplayer1.append(8)
            #print(arrplayer1)

            balcycle1 = 0
            arrchoice1 = []
            while balcycle1 < numberforcycles:
              ballnow = self.fields[balcycle1][2]
              ballindex = self.fields[balcycle1][0]
              if ballnow != 0:
                ballfuture = ballindex + roll
                if ballfuture <= numberforcycles:
                 if ballfuture not in arrplayer1:
                  arrchoice1.append(ballfuture)
              # print("Выпала" , self.roll)
                  #print("текущая позиция" , ballindex)
                  #print("возможная позиция" , ballfuture)
              balcycle1 += 1



            if arrchoice1:

              #случайный бот:
              #playeronemove = random.choice(arrchoice1)

              #Очень жадный бот:
              #playeronemove = len(arrchoice1)
              #playeronemove = arrchoice1[playeronemove-1]

              #Жадный бот:
              playeronemove = 33
              for arrchoice in arrchoice1:
                if self.fields[arrchoice][1] == 1 or (self.fields[arrchoice][3] == 1 and self.fields[arrchoice][0] >= 5 and self.fields[arrchoice][0] <= 12):
                  playeronemove = self.fields[arrchoice][0]
              if playeronemove == 33:
                playeronemove = len(arrchoice1)
                playeronemove = arrchoice1[playeronemove-1]

              #playeronemove = random.choice(arrchoice1)
              playeronepos = playeronemove - roll
              #print(playeronepos)
              self.fields[playeronepos][2] = self.fields[playeronepos][2] - 1
              self.fields[playeronemove][2] = self.fields[playeronemove][2] + 1
              if self.fields[playeronemove][3] == 1 and playeronemove >= 5 and playeronemove <= 12:
                self.fields[playeronemove][3] = 0
                self.fields[0][3] = self.fields[0][3] + 1
                #print("мы забрали шашку на" , self.fields[playeronemove][0])
              #print("боту выпал", roll)
              #print("бот ходит", playeronemove)
              return playeronemove

     def second():
            numberforcycles = 15

            agentroll = self.fields[16][1]


            lst = [0, 1, 2, 3, 4]
            weights = [6.25, 25, 37.5, 25, 6.25]
            nextagentroll = random.choices(lst, weights=weights, k=1)
            nextagentroll = nextagentroll[0]
            self.fields[16][1] = nextagentroll
            #print("Агенту Выпал" , agentroll)

            agentcycle = 0
            arrplayeragent = []

            while agentcycle < numberforcycles:
              agentballnow = self.fields[agentcycle][3]
              agentballindex = self.fields[agentcycle][0]
          #расположение фишек игрока 2
              if agentballnow != 0:
               arrplayeragent.append(agentballindex)
              agentcycle += 1
            if self.fields[8][2] == 1:
                arrplayeragent.append(8)

            #print("фишки агента", arrplayeragent)

            agentcycle = 0
            arrplayeragent2 = []

            while agentcycle < numberforcycles:
              agentballnow = self.fields[agentcycle][3]
              agentballindex = self.fields[agentcycle][0]
          #будущее фишек игрока 2
              if agentballnow != 0:
               agentballfuture = agentballindex + agentroll
               if agentballfuture <= numberforcycles:
                if agentballfuture not in arrplayeragent:
                  arrplayeragent2.append(agentballfuture)
              agentcycle += 1

            #print(arrplayeragent2)



            if arrplayeragent2:
              playeragentmove = len(arrplayeragent2)
              playeragentmove = arrplayeragent2[0]#arrplayeragent2[playeragentmove-1]

              if action == self.FIRST:
                if len(arrplayeragent2) > 0:
                 playeragentmove = arrplayeragent2[0]
                else:
                 playeragentmove = playeragentmove
              elif action == self.SECOND:
               if len(arrplayeragent2) > 1:
                playeragentmove = arrplayeragent2[1]
               else:
                playeragentmove = playeragentmove
              elif action == self.THIRD:
               if len(arrplayeragent2) > 2:
                playeragentmove = arrplayeragent2[2]
               else:
                playeragentmove = playeragentmove
              elif action == self.FORTH:
               if len(arrplayeragent2) > 3:
                playeragentmove = arrplayeragent2[3]
               else:
                playeragentmove = playeragentmove
              elif action == self.FIFTH:
               if len(arrplayeragent2) > 4:
                playeragentmove = arrplayeragent2[4]
               else:
                playeragentmove = playeragentmove
              elif action == self.SIXTH:
               if len(arrplayeragent2) > 5:
                playeragentmove = arrplayeragent2[5]
               else:
                playeragentmove = playeragentmove
              elif action == self.SEVENTH:
               if len(arrplayeragent2) > 6:
                playeragentmove = arrplayeragent2[6]
               else:
                playeragentmove = playeragentmove
              else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(action))



              #print("Агент делает выбор", playeragentmove)

              #playeragentmove = random.choice(arrplayeragent)
              playeragentpos = playeragentmove - agentroll
              self.fields[playeragentpos][3] = self.fields[playeragentpos][3] - 1
              self.fields[playeragentmove][3] = self.fields[playeragentmove][3] + 1
              if self.fields[playeragentmove][2] == 1 and playeragentmove >= 5 and playeragentmove <= 12:
                self.fields[playeragentmove][2] = 0
                self.fields[0][2] = self.fields[0][2] + 1
                #print("Агент забрал шашку на" , self.fields[playeragentmove][0])
              #print("Агенту выпал" , agentroll)
              #print("Агент ходит" , playeragentmove)
              return playeragentmove
            else:
              rrr = 0#print("Выпал 0 и агент не делает выбора")
     #while True:
     # thisbotmove = first()
     # if thisbotmove != 4 or thisbotmove != 8 or thisbotmove != 14:
     #    #print("нет второго хода бота")
     #    break

     if self.fields[16][3] != 1:
      thisbotmove = first()
      #print(first())
      if thisbotmove == 4 or thisbotmove == 8 or thisbotmove == 14:
       #print("Второй ход бота", thisbotmove)
       self.fields[16][2] = 1
      else:
       self.fields[16][2] = 0
    # if thisbotmove == 4 or thisbotmove == 8 or thisbotmove == 14:
     if self.fields[16][2] != 1:
      thisagentmove = second()
      #print(second())
      if thisagentmove == 4 or thisagentmove == 8 or thisagentmove == 14:
        #print("Второй ход агента", thisagentmove)
        self.fields[16][3] = 1
      else:
        self.fields[16][3] = 0
     #while True:
     # thisagentmove =  second()
     # if thisagentmove != 4 or thisagentmove != 8 or thisagentmove != 14:
     #    #print("нет второго хода агента")
     #    break


     terminated = bool(self.fields[15][2] >= 7 or self.fields[15][3] >= 7)
     truncated = False  # we do not limit the number of steps here

     reward = 0
     if self.fields[15][3] >= 7 and self.fields[15][2] != 7:
       reward = 100

    # Optionally we can pass additional info, we are not using that for now
     info = {}

     return np.array(self.fields).astype(np.int8), reward, terminated, truncated, info


  def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            1==1

    #print(self.fields[0])

  def close(self):
    pass
  

from stable_baselines3 import PPO, A2C, DQN

env = GoLeftEnv(grid_size=10)

model = PPO("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2500000), progress_bar=True)

iii=0
iiia = []
env = GoLeftEnv()
obs = env.reset()
while iii < 50:
 n_steps = 100
 vec_env = model.get_env()
 obs = vec_env.reset()
 for i in range(n_steps):
     action, _states = model.predict(obs, deterministic=True)
     obs, rewards, dones, info = vec_env.step(action)
     print("Action: ", action)
     print('obs=', obs, 'rewards=', rewards, 'dones=', dones)
     print("observation space shape:", env.observation_space.shape)
     if obs[0][4][2] == 1:
       print(u"\U0001F7E5", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][5][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][5][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][4][3] == 1:
       print(u"\U0001F7E6")
     else:
       print(u"\u2B1C")
     if obs[0][3][2] == 1:
       print(u"\U0001F7E5", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][6][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][6][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][3][3] == 1:
       print(u"\U0001F7E6")
     else:
       print(u"\u2B1C")
     if obs[0][2][2] == 1:
       print(u"\U0001F7E5", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][7][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][7][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][2][3] == 1:
       print(u"\U0001F7E6")
     else:
       print(u"\u2B1C")
     if obs[0][1][2] == 1:
       print(u"\U0001F7E5", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][8][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][8][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][1][3] == 1:
       print(u"\U0001F7E6")
     else:
       print(u"\u2B1C")
     print(u"\u2B1B", end="")
     if obs[0][9][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][9][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     print(u"\u2B1B")
     print(u"\u2B1B", end="")
     if obs[0][10][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][10][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     print(u"\u2B1B")
     if obs[0][14][2] == 1:
       print(u"\U0001F7E5", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][11][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][11][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][14][3] == 1:
       print(u"\U0001F7E6")
     else:
       print(u"\u2B1C")
     if obs[0][13][2] == 1:
       print(u"\U0001F7E5", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][12][2] == 1:
       print(u"\U0001F7E5", end="")
     elif obs[0][12][3] == 1:
       print(u"\U0001F7E6", end="")
     else:
       print(u"\u2B1C", end="")
     if obs[0][13][3] == 1:
       print(u"\U0001F7E6")
     else:
       print(u"\u2B1C")
     vec_env.render()
     if dones:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", rewards)
        iiia.append(rewards)
        break
 iii += 1
print(iiia)
win = []
notwin = []
for iiiaa in iiia:
   if iiiaa <= 0:
    notwin.append(iiiaa)
   elif iiiaa > 0:
    win.append(iiiaa)
print(win)
print(notwin)
print(len(win))
print(len(notwin))

# sample action:
print("sample action:", env.action_space.sample())

# observation space shape:
print("observation space shape:", env.observation_space.shape)