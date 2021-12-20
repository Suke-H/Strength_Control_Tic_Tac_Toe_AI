from math import inf
import numpy as np
import time
import random
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Human:
  ''' input = current state => update GUI, output = mouse click => move + update GUI'''
  def __init__(self):
    import gui           # On importe gui ici seulement car on a pas besoin de pygame si les IAs jouent entre eux
    self.gui = gui.gui() # On initialise le gui ici car move() est la 1ère methode à être appellée

  def move(self,board,turn):       # Renvoie sous forme d'un tuple (row,col) le coup du joueur
    self.board = board
    self.update()

    move = self.decision()

    self.board[move[0],move[1]] = turn
    self.update()
    return move

  def update(self):     # Actualise l'écran
    self.gui.draw_board()
    self.gui.draw_xo(self.board)

  def decision(self):   # Renvoie les la case que l'humain choisi
    return self.gui.play()

  def show_end_state(self,board): # Affiche l'état final avec une ligne si quelqu'un a gagné
    self.board = board
    self.update()
    self.gui.draw_line(board)
    time.sleep(1)

class Random:
  ''' input = board, output = random move'''

  def move(self,board,turn):
    self.board = board
    possible_moves = np.argwhere(self.board == 0)   # On fait une liste de tous les indices avec une valeur de 0
    move = np.random.permutation(possible_moves)[0] # On permute aléatoirement l'array et on prends le premier indice

    return (move[0],move[1])                    # On le retourne sous la forme d'un tuple

############## ALGORITHMES ##############

def win_eval(board):      # Je réécris la fonction pour éviter une import loop
  ''' Fonction qui évalue une grille et renvoie : 0 si la partie n'est pas finie, 1 si O gagne, 2 si X gagne ou 3 si match nul '''
  
  board = np.array(board) if type(board) == list else board   # Si la grille venait a être une liste (minimax utilise des listes)

  for x in range(3):
    if board[x,0] == board[x,1] == board[x,2] != 0: return board[x,0]    # -
    elif board[0,x] == board[1,x] == board[2,x] != 0: return board[0,x]  # |
    elif board[0,0] == board[1,1] == board[2,2] != 0: return board[0,0]  # \
    elif board[0,2] == board[1,1] == board[2,0] != 0: return board[0,2]  # /
  if np.count_nonzero(board) < 9: return 0  # Pas fini
  return 3                                  # Match nul

def score_eval(board,turn):
  ''' Calcule le score à donner à l'algorithme '''
  antiturn = 0
  if turn == 1: 
    antiturn = 2
  elif turn == 2: 
    antiturn = 1
    
  if win_eval(board) == antiturn:   # Si l'autre gagne
    score = -1
  elif win_eval(board) == turn: # Si minimax gagne
    score = 1
  else:
    score = 0
  return score

class Q_learning:
  ''' input = current state, output = new move '''
  def __init__(self,Q={},epsilon=0.3, alpha=0.2, gamma=0.9):
    self.q_table = Q
    self.epsilon = epsilon    # Exploration vs Exploitation
    self.alpha = alpha          # Learning rate
    self.gamma = gamma          # Discounting factor

  def encode(self,state):      # Encode array to string
    s = ''
    for row in range(3):
      for col in range(3):
        s += str(state[row,col])
    return s

  def decode(self,s):          # Decode string to array
    return np.array([[int(s[0]),int(s[1]),int(s[2])],[int(s[3]),int(s[4]),int(s[5])],[int(s[6]),int(s[7]),int(s[8])]])

  def format(self,action):        # Convert any tuple to int
    if type(action) == int:
      return action
    else:
      return 3*action[0] + action[1]

  def possible_actions(self,board):
    ''' retourne tous les indices de valeur 0 '''
    return [i for i in range(9) if self.encode(np.array(board))[i]=='0']

  def q(self,state,action):
    action = self.format(action)
    if (self.encode(state),action) not in self.q_table:
      self.q_table[(self.encode(state),action)] = 1    # On est optimiste pour encourager l'exploration
    return self.q_table[(self.encode(state),action)]

  def move(self,board,turn):
    self.board = board
    actions = self.possible_actions(board)
    
    if random.random() < self.epsilon:        # exploration
      self.last_move = random.choice(actions)
      self.last_move = (self.last_move//3,self.last_move%3) # on retourne le move sous forme de tuple
      return self.last_move
    
    # else: exploitation
    q_values = [self.q(self.board, a) for a in actions]
    
    if turn == 2:   # Si q_learning joue X
      max_q = max(q_values)
    else:           # Si q_learning joue O
      max_q = min(q_values)

    if q_values.count(max_q) > 1:       # s'il y a plusieurs max_q, choisir aléatoirement
      best_actions = [i for i in range(len(actions)) if q_values[i] == max_q]
      i = np.random.permutation(best_actions)[0]
    else:
      i = q_values.index(max_q)

    self.last_move = actions[i]
    self.last_move = (self.last_move//3,self.last_move%3)
    return self.last_move

  def learn(self,S,A,S1,A1,reward):
    A = self.format(A)
    A1 = self.format(A1)

    prev = self.q(S,A)
    maxnewq = self.q(S1,A1)
    
    S = self.encode(S)
    S1 = self.encode(S1)

    self.q_table[(S,A)] = prev + self.alpha * (reward + self.gamma*maxnewq - prev)
    
class DQNNet(torch.nn.Module):
    
    def __init__(self):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.net(x)
        return x.tanh()
    
# class DQNNet(nn.Module):
#     def __init__(self):
#         # model is initiated in parent class, set params early.
#         self.obs_size = 9
#         self.n_actions = 9
#         super(DQNNet, self).__init__()

#     def model(self):
#         # observations -> hidden layer with relu activation -> actions
#         return nn.Sequential(
#             nn.Linear(self.obs_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.n_actions)
        # )
    
class DQN:
  ''' input = current state, output = new move '''
  def __init__(self,Q={},epsilon=0.3, alpha=0.2, gamma=0.9):
    self.q_table = Q
    self.epsilon = epsilon    # Exploration vs Exploitation
    self.alpha = alpha          # Learning rate
    self.gamma = gamma          # Discounting factor
    
    self.minibatch_size = 256
    self.replay_memory_size = 131072
    self.discount_factor = 0.99
    self.epsilon = 0.1
    
    # replay memory
    self.D = deque(maxlen=self.replay_memory_size)

    # model
    self.model = DQNNet().to(device)
    self.criterion = nn.MSELoss()
    self.optimizer = optim.RMSprop(self.model.parameters(), lr=10**(-5))

    # for log
    self.current_loss = 0.0

  def encode(self,state):      # Encode array to string
    s = ''
    for row in range(3):
      for col in range(3):
        s += str(state[row,col])
    return s

  def decode(self,s):          # Decode string to array
    return np.array([[int(s[0]),int(s[1]),int(s[2])],[int(s[3]),int(s[4]),int(s[5])],[int(s[6]),int(s[7]),int(s[8])]])

  def format(self,action):        # Convert any tuple to int
    if type(action) == int:
      return action
    else:
      return 3*action[0] + action[1]

  def possible_actions(self,board):
    ''' retourne tous les indices de valeur 0 '''
    return [i for i in range(9) if self.encode(np.array(board))[i]=='0']

  def q(self,state,action):
    """ return q(s, a) """
  
    # x = self.encode(state)
    # x = np.array([int(s) for s in x]).astype(np.float32)

    if len(state.shape) == 1:
          state = state.reshape(1, len(state))
    x = torch.from_numpy(state)
    x = x.to(device)
    outputs = self.model(x)

    return outputs[0, action].item()

  def move(self,board,turn):
    self.board = board
    actions = self.possible_actions(board)
    
    if random.random() < self.epsilon:        # exploration
      self.last_move = random.choice(actions)
      self.last_move = (self.last_move//3,self.last_move%3) # on retourne le move sous forme de tuple
      return self.last_move
    
    # else: exploitation
    x = self.encode(self.board)
    x = np.array([int(s) for s in x]).astype(np.float32)
    q_values = [self.q(x, a) for a in actions]
    
    if turn == 2:   # Si q_learning joue X
      max_q = max(q_values)
    else:           # Si q_learning joue O
      max_q = min(q_values)

    if q_values.count(max_q) > 1:       # s'il y a plusieurs max_q, choisir aléatoirement
      best_actions = [i for i in range(len(actions)) if q_values[i] == max_q]
      i = np.random.permutation(best_actions)[0]
    else:
      i = q_values.index(max_q)

    self.last_move = actions[i]
    self.last_move = (self.last_move//3,self.last_move%3)
    return self.last_move
  
  def train_model(self, x, y, n_epochs):
    # stateをCNN用に変換
    # x = self.encode(x)
    # x = np.array([int(s) for s in x]).astype(np.float32)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x, y = x.to(device), y.to(device)
    
    for i in range(n_epochs):
      self.optimizer.zero_grad()
      outputs = self.model(x)
      loss = self.criterion(outputs, y)
      loss.backward()
      self.optimizer.step()

    return loss
  
  def store_experience(self, state, action, reward, state_1, terminal):
    # self.D.append((state_transform(state), action, reward, state_transform(state_1), terminal))
    self.D.append((state, action, reward, state_1, terminal))

  def experience_replay(self):
    state_minibatch = []
    y_minibatch = []

    # ミニバッチサイズ(Dがミニバッチサイズ分たまってなかったらDの長さ)
    minibatch_size = min(len(self.D), self.minibatch_size)
    # Dからミニバッチをランダムに選んで作成
    minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

    for j in minibatch_indexes:
      state_j, action_j, reward_j, state_j_1, terminal = self.D[j]

      y_j = np.array([self.q(state_j, a) for a in range(9)])
      # y_j = self.Q_values(state_j)

      if terminal:
          y_j[action_j] = reward_j
      else:
          # y = reward + gamma * Q(state', max_action')
          Q_j_1 = [self.q(state_j_1, a) for a in range(9)]
          y_j[action_j] = reward_j + self.discount_factor * max(Q_j_1)

      state_minibatch.append(state_j)
      y_minibatch.append(y_j)

    # 学習
    state_minibatch, y_minibatch = np.array(state_minibatch).astype(np.float32), np.array(y_minibatch).astype(np.float32)
    # state_minibatch = state_minibatch.reshape(state_minibatch.shape[0], 64)

    self.current_loss = self.train_model(state_minibatch, y_minibatch, 50)
