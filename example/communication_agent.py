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
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.models import model_from_json
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

file = open('loss.txt', 'a')

max_score = 0
n_episodes = 5000
n_win_player = 1000
max_env_step = 1000

gamma = 1.0
epsilon = 0.5
#epsilon_min = 0.01
#epsilon_decay = 0.999

alpha = 0.001 # learning rate
#alpha_decay = 0.01
alpha_test_factor = 1.0

batch_size = 512

memory = deque(maxlen=100000)
hfo = HFOEnvironment()

feature_size = 5 # 12
out_layer = 36

round = 0

min_loss = 9999.0


model = Sequential()
model.add(Dense(12,input_dim=feature_size,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(out_layer,activation='relu'))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


model.compile(loss='mse', optimizer=adam(lr=alpha), metrics=['accuracy'])



def filter_data(state):
  return state[0:5]
def calc_reward(last_state,state):
  # if state[5] == True:
  #   return 100.0
  self_x = state[0] * 52.5
  self_y = state[1] * 34.0
  ball_x = state[3] * 52.5
  ball_y = state[4] * 34.0
  dist = math.sqrt(pow(self_x - ball_x , 2) + pow(self_y - ball_y, 2))
  last_self_x = last_state[0] * 52.5
  last_self_y = last_state[1] * 34.0
  last_ball_x = last_state[3] * 52.5
  last_ball_y = last_state[4] * 34.0
  last_dist = math.sqrt(pow(last_self_x - last_ball_x, 2) + pow(last_self_y - last_ball_y, 2))

  return 100*(last_dist - dist)


  # if state[5] == True:
  #   return 10.0
  # else:
  #   return 0.0;
def create_sample_action():
  dash_param = []
  for a in range(36):
    #for p in range(10):
    temp_act = [100,a*10]
    dash_param.append(temp_act)
  return dash_param
def remember(state, action, reward, next_state, status):
  memory.append((state, action, reward, next_state, status))
def choose_action(state, epsilon):
  global round
  acts = create_sample_action()
  round += 1
  if round >= 2000:
    round = 0
  if round >= 1000:
    index = np.argmax(model.predict(state))
    return acts[index]
  if np.random.random() <= epsilon:
    max_index = len(acts)
    index = random.randint(0, max_index - 1)
    print("rand : " + str(acts[index][1]))
    return acts[index]
  else:
    index =  np.argmax(model.predict(state))
    print("best " + str(acts[index][1]))
    return acts[index]
def get_epsilon(t):
  global epsilon
  return epsilon#max(epsilon_min, min(epsilon , 1.0 - math.log10((t+1)*epsilon_decay)))
def preprocess(state):
  return np.reshape(state, [1,feature_size])
def replay(batch_size, epsilon):
  global min_loss
  x_batch, y_batch = [] , []
  minibatch = random.sample(memory, min(len(memory), batch_size))
  for state, action, reward, next_state, status in minibatch:
    y_target = model.predict(state)
    #print(y_target)
    index = action[1]/10
    y_target[0][int(index)] = reward# + gamma * np.max(model.predict(next_state)[0])
    x_batch.append(state[0])
    y_batch.append(y_target[0])
  hist = model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0, epochs=10, callbacks=[tensorboard] )
  loss = hist.history['loss'][len(hist.history['loss'])-1]
  cur_loss = hist.history['loss'][len(hist.history['loss'])-1]
  print("LOSS : "  + str(cur_loss))
  file.write(str(cur_loss) + "\n")
  if cur_loss < min_loss:
    min_loss = cur_loss
    model.save_weights('final_model_weights.h5')
    with open('final_model_architecture.json', 'w') as f:
      f.write(model.to_json())

def main():
  global max_score
  scores = deque(maxlen=100)



  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      '../bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)



  for episode in itertools.count():


    status = IN_GAME


    while status == IN_GAME:
      features = hfo.getState()
      usefull_feateures = filter_data(features)
      state = preprocess(usefull_feateures)
      action = choose_action(state, get_epsilon(episode))
      hfo.act(DASH, action[0], action[1]) # first param : type - for dash : second : power for dash third : angle
      status = hfo.step()
      if status != IN_GAME:
        break
      next_features = hfo.getState()
      next_usefull_features = filter_data(next_features)
      reward = calc_reward(usefull_feateures,next_usefull_features)
      print("REWARD : " + str(reward))
      next_state = preprocess(next_usefull_features)
      remember(state, action , reward, next_state, status)
      replay(batch_size, get_epsilon(episode))



  if status == SERVER_DOWN:
    hfo.act(QUIT)
    file.close()
    exit()

if __name__ == '__main__':
  main()
