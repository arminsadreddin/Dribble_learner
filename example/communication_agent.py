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
from sympy import *
from sympy.geometry import *
from keras import optimizers
file = open('loss.txt', 'a')

max_score = 0
n_episodes = 5000
n_win_player = 1000
max_env_step = 1000

gamma = 0.9
epsilon = 0.1
state_stack = []
#epsilon_min = 0.01
#epsilon_decay = 0.999

alpha = 0.01 # learning rate
#alpha_decay = 0.01
alpha_test_factor = 1.0

batch_size = 100000

memory = deque(maxlen=100000)
hfo = HFOEnvironment()

feature_size = 6 # 12
out_layer = 72

round = 0

min_loss = 9999.0
def load_model():
  # Model reconstruction from JSON file
  with open('final_model_architecture_v8.json', 'r') as f:
    model = model_from_json(f.read())

  # Load weights into the new model
  model.load_weights('final_model_weights_v8.h5')
  return model

model = load_model()
# model = Sequential()
# model.add(Dense(12,input_dim=feature_size,activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(48,activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(out_layer,activation='linear'))

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=alpha, nesterov=True), metrics=['accuracy'])
model.summary()



def is_final_state(state):
  return state[5]
def filter_data(state):
  data = []
  data.append(state[0])
  data.append(state[1])
  data.append(state[2])
  data.append(state[3])
  data.append(state[4])
  data.append(state[len(state) - 1 ])
  return data
def calc_reward(last_state,state):
  step_cost = -0.01
  ball_x = (state[3] + 1) * 52.5
  ball_y = (state[4] + 1) * 34.0
  last_self_x = (last_state[0] + 1) * 52.5
  last_self_y = (last_state[1] + 1) * 34.0
  self_x = (state[0] + 1) * 52.5
  self_y = (state[1] + 1) * 34.0
  #return (300.0-sqrt(pow(self_x-ball_x,2)+pow(self_y-ball_y,2))) - 0.1


  target = Point(ball_x , ball_y)
  cur_self = Point(self_x , self_y)
  last_self = Point(last_self_x , last_self_y)
  last_to_target = Line(last_self , target)
  if cur_self == last_self:
    return step_cost
  cur_to_target = Line(last_self , cur_self)
  dir_value = float(math.cos(cur_to_target.angle_between(last_to_target))*10.0)
  dist_value = float(cur_self.distance(last_self))
  reward = float(dir_value * dist_value + step_cost)
  return reward

  # if state[5] == True:
  #   return 100.0
  #
  # # if state[5] == True:
  # #   return 100.0
  # self_x = state[0] * 52.5
  # self_y = state[1] * 34.0
  # ball_x = state[3] * 52.5
  # ball_y = state[4] * 34.0
  # dist = math.sqrt(pow(self_x - ball_x , 2) + pow(self_y - ball_y, 2))
  # last_self_x = last_state[0] * 52.5
  # last_self_y = last_state[1] * 34.0
  # last_ball_x = last_state[3] * 52.5
  # last_ball_y = last_state[4] * 34.0
  # last_dist = math.sqrt(pow(last_self_x - last_ball_x, 2) + pow(last_self_y - last_ball_y, 2))
  #
  # return 10*(last_dist - dist)



  # else:
  #   return 0.0;
def create_sample_action():
  dash_param = []
  for a in range(36):
    temp_act = ["DASH",100,a*10]
    dash_param.append(temp_act)
  for a in range(36):
    temp_act=["TURN",a*10,""]
    dash_param.append(temp_act)
  return dash_param
def remember(state, action, reward, next_state, status):
  memory.append((state, action, reward, next_state, status))
def choose_action(state, epsilon):
  global round
  acts = create_sample_action()
  #round += 1
  #if round >= 6000:
  #   round = 0
  #if round >= 5000:
  #index = np.argmax(model.predict(state))
  #return acts[index]
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
    index = 0
    if action[0] == "DASH":
      index = action[2] / 10
    elif action[0] == "TURN":
      index = action[1] / 10
      index += 36
    y_target[0][int(index)] = reward + gamma * np.max(model.predict(next_state)[0])
    x_batch.append(state[0])
    y_batch.append(y_target[0])
  hist = model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0, epochs=1)
  loss = hist.history['loss'][len(hist.history['loss'])-1]
  cur_loss = hist.history['loss'][len(hist.history['loss'])-1]
  print("LOSS : "  + str(cur_loss))
  file.write(str(cur_loss) + "\n")
  if cur_loss < min_loss and len(memory) >= 1000 :
    min_loss = cur_loss
    model.save_weights('final_model_weights_v8.h5')
    with open('final_model_architecture_v8.json', 'w') as f:
      f.write(model.to_json())

def main():
  global max_score
  global hfo
  scores = deque(maxlen=100)
  global alpha

  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      '../bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)



  for episode in itertools.count():


    status = IN_GAME

    cur_state = hfo.getState()
    hfo.act(NOOP)
    hfo.step()
    while status == IN_GAME and is_final_state(cur_state) == -1 and ( (cur_state[0] < 1 and cur_state[0] > -1) and (cur_state[1] < 1 and cur_state[1] > -1) ):
      cur_state = hfo.getState()
      usefull_feateures = filter_data(cur_state)
      state = preprocess(usefull_feateures)
      action = choose_action(state, get_epsilon(episode))
      if action[0] == "DASH":
        hfo.act(DASH, action[1], action[2])  # first param : type - for dash : second : power for dash third : angle
      elif action[0] == "TURN":
        hfo.act(TURN, action[1])  # first param : type - for dash : second : power for dash third : angle
      status = hfo.step()
      if status != IN_GAME or ( (cur_state[0] > 1 or cur_state[0] < -1) or (cur_state[1] > 1 or cur_state[1] < -1) ):
        break
      cur_state = hfo.getState()
      next_usefull_features = filter_data(cur_state)
      reward = calc_reward(usefull_feateures,next_usefull_features)
      print("REWARD : " + str(reward))
      next_state = preprocess(next_usefull_features)
      remember(state, action , reward, next_state, status)
      if len(memory) >= 2000:
        replay(batch_size, get_epsilon(episode))


  if status == SERVER_DOWN:
    hfo.act(QUIT)
    file.close()
    exit()

if __name__ == '__main__':
  main()
