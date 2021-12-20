from math import inf
import numpy as np
import time
import random

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

def minimax_algorithm(board,depth,turn,maximizingPlayer=True):
  ''' Source : https://fr.wikipedia.org/wiki/Algorithme_minimax
      Cet algorithme va explorer l'arbre de tous les coups possible '''
  board = list(board)   # Pour des raisons que j'ignore, ça ne marche pas avec des arrays numpy

  if win_eval(board) != 0:     # Si on se trouve dans un état final
    return score_eval(board,turn)
  if maximizingPlayer:
    best_score = -inf
    for row in range(3):
      for col in range(3):
        if board[row][col] == 0:   # Si la case est vide
          board[row][col] = 1      # Comme minimac est le maximizingPlayer et qu'il joue O, on essaye pour chaque case vide de mettre un O 
          score = minimax_algorithm(board,depth-1,turn,False)     # On calcul le score récursivement
          board[row][col] = 0      # On remet la valeur initiale (ça nous évite de devoir deep copy la liste)
          best_score = max(best_score,score)        # On garde le score maximal
    return best_score
  else:  # On fait exactement la même chose mais avec min(), on suppose que l'adversaire va toujours jouer le meilleur coup possible
    best_score = +inf
    for row in range(3):
      for col in range(3):
        if board[row][col] == 0:
          board[row][col] = 2
          score = minimax_algorithm(board,depth-1,turn,True)
          board[row][col] = 0
          best_score = min(best_score,score)
    return best_score    

def best_move(board,turn):
  ''' Fonction qui va déterminer le meilleur coup à jouer pour minimax selon la grille '''
  best_score = -inf
  for row in range(3):
    for col in range(3):
      if board[row,col] == 0:
        board[row,col] = turn

        score = minimax_algorithm(board,list(np.ravel(board)).count(0),turn,False)   # Pour avoir la depth, on compte le nombre de 0 dans la grille
        
        board[row,col] = 0
        best_score = max(best_score,score)
        if best_score == score:
          move = (row,col)
  return move

class Minimax:
  ''' input = current state, output = new move ''' 
  def move(self,board,turn):
    return best_move(board,turn)


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