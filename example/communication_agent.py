#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import sys, itertools
from hfo import *
import math
import keras
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam

max_score = 0
n_episodes = 5000
n_win_player = 1000
max_env_step = 1000

gamma = 1.0
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

alpha = 0.01 # learning rate
alpha_decay = 0.01
alpha_test_factor = 1.0

batch_size = 256
monitor = False
quiet= False

memory = deque(maxlen=100000)
hfo = HFOEnvironment()

feature_size = 12

model = Sequential()
model.add(Dense(12,input_dim=feature_size,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(48,activation='relu'))
model.add(Dense(96,activation='relu'))
model.add(Dense(192,activation='relu'))
model.add(Dense(360,activation='relu'))
model.compile(loss='mse', optimizer=adam(lr=alpha, decay=alpha_decay))

def calc_reward(state):
  if state[5] == True:
    return 10.0
  else:
    return 0.0;
def create_sample_action():
  dash_param = []
  for a in range(36):
    for p in range(10):
      temp_act = [p*10,a*10]
      dash_param.append(temp_act)
  return dash_param
def remember(state, action, reward, next_state, done):
  memory.append((state, action, reward, next_state, done))
def choose_action(state, epsilon):
  acts = create_sample_action()
  if np.random.random() <= epsilon:
    max_index = len(acts)
    index = random.randint(0, max_index - 1)
    print("rand")
    return acts[index]
  else:
    print("best")
    index =  np.argmax(model.predict(state))
    return acts[index]
def get_epsilon(t):
  return max(epsilon_min, min(epsilon , 1.0 - math.log10((t+1)*epsilon_decay)))
def preprocess(state):
  return np.reshape(state, [1,12])
def replay(batch_size, epsilon):
  x_batch, y_batch = [] , []
  minibatch = random.sample(memory, min(len(memory), batch_size))
  for state, action, reward, next_state, status in minibatch:
    y_target = model.predict(state)
    y_target[0][action] = reward + gamma * np.max(model.predict(next_state)[0])
    x_batch.append(state[0])
    y_batch.append(y_target[0])
  model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
  if epsilon > epsilon_min:
    epsilon *= epsilon_decay
def main():
  global max_score
  scores = deque(maxlen=100)
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      '../bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)
  for episode in itertools.count():
    # if episode > itertools.count() - 2:
    #   global epsilon
    #   epsilon = 0.0

    status = IN_GAME
    i = 0.0
    hfo.act(REORIENT)
    while True:
      features = hfo.getState()
      state = preprocess(features)
      action = choose_action(state, get_epsilon(episode))
      #print(action)
      #print(action[0])
      #print(action[1])
      hfo.act(DASH, action[0], action[1]) # first param : type - for dash : second : power for dash third : angle
      status = hfo.step()
      next_features = hfo.getState()
      reward = calc_reward(next_features)
      next_state = preprocess(next_features)
      remember(state, action , reward, next_state, status)
      i += reward
      #print("REWARD : " + str(reward))
      # if reward != 0.0:
      #   print("-------------------- WHILE BREAK ------------------------------")
      #   break
      if reward != 0:
        print(" - - - - - - - - - -  - got reward - - - - - - - - - - - - - - ")
    if i > max_score:
      max_score = i
    scores.append(i)
    mean_score = np.mean(scores)
    replay(batch_size, get_epsilon(episode))
    print(mean_score)
  print(max_score)


  if status == SERVER_DOWN:
    hfo.act(QUIT)
    exit()

if __name__ == '__main__':
  main()
