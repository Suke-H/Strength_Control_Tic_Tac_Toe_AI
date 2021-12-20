# Tic Tac Toe Reinforcement learning
![Game GUI](https://i.imgur.com/gjPgo0E.png)
## Setup
Install requirements with pip

```bash
pip install -r requirements.txt
```
## Run
```bash
python main.py
```
### Change agents
Example :
```python
if __name__ == "__main__":
  X_player = agents.Random()
  O_player = agents.Minimax()
  print(game(X_player,O_player,True))
```
### Q-learning
Train over 10k iterations to get good results, 200k to get near perfect results.

When trained playing 200'000 games as X against random, the Q-learning algorithm achieves 99.99% win-rate , 0.01% draw-rate and 0% lose-rate against the random player.

![Plotted results](https://i.imgur.com/sCIFTEU.png)

PS: The code is commented in french.
