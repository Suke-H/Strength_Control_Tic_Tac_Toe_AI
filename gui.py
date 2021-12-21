from main import win_eval
# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame,sys
import time

pygame.init()
screen = pygame.display.set_mode((600,600))
screen.fill((0,0,0))

def board_pos(x,y,size=200):
  ''' Prends des coordonées (x,y) en paramètres et renvoie la ligne et la colonne '''
  for a in range(1,4):
    if x < size*a:
      col = a-1
      break
  for a in range(1,4):
    if y < size*a:
      row = a-1
      break
  return (row,col)

class gui:
  def __init__(self):
    self.display = pygame.display.set_mode((600,600))
    self.color = (255,255,255)
    self.size = 200

  def draw_board(self):
    ''' Dessine une grille 3x3 vide '''
    for row in range(3):
      for col in range(3):
        pygame.draw.rect(self.display, self.color, (row*self.size,col*self.size,self.size,self.size),5)
    pygame.display.update()

  def draw_xo(self,board):
    ''' Dessine les X et O à partir d'une grille donnée sous la forme d'une matrice numpy (array) '''
    for row in range(3):
      for col in range(3):
        x_center = int(col*self.size + self.size/2)
        y_center = int(row*self.size + self.size/2)
        if board[row,col] == 1:      # O
          pygame.draw.circle(self.display,self.color,(x_center,y_center),75,5) # Dessine un cercle au centre de la case
        elif board[row,col] == 2:    # X
          pygame.draw.line(self.display,self.color,(x_center-75,y_center-75),(x_center+75,y_center+75),5) # Dessine 2 traits pour
          pygame.draw.line(self.display,self.color,(x_center-75,y_center+75),(x_center+75,y_center-75),5) # former un X
    pygame.display.update()

  def draw_line(self,board):
    ''' dessine une ligne si 3 cases se suivent '''
    for x in range(3):
      if board[x,0] == board[x,1] == board[x,2] != 0: # -
        pygame.draw.line(self.display,self.color,
                        (0*self.size + self.size/2, x*self.size + self.size/2),
                        (2*self.size + self.size/2, x*self.size + self.size/2),5)

      elif board[0,x] == board[1,x] == board[2,x] != 0: # |
        pygame.draw.line(self.display,self.color,
                        (x*self.size + self.size/2, 0*self.size + self.size/2),
                        (x*self.size + self.size/2, 2*self.size + self.size/2),5)

      elif board[0,0] == board[1,1] == board[2,2] != 0: # \
        pygame.draw.line(self.display,self.color,
                        (0*self.size + self.size/2, 0*self.size + self.size/2),
                        (2*self.size + self.size/2, 2*self.size + self.size/2),5)

      elif board[0,2] == board[1,1] == board[2,0] != 0: # /
        pygame.draw.line(self.display,self.color,
                        (2*self.size + self.size/2, 0*self.size + self.size/2),
                        (0*self.size + self.size/2, 2*self.size + self.size/2),5)

    pygame.display.update()

  def play(self):
    ''' Prends en compte les clicks le la souris '''
    while True:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          print("un?")
          sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:  # Si le boutton de la souris est pressé
            (x,y) = pygame.mouse.get_pos()
            return board_pos(x,y,self.size)
    time.sleep(0.05)
            
