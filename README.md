# Steam Platform Analysis

## Methodology
To identify potential Nexon players, we created a “Nexon-profile” based on gameplay and purchase history. If a player fits the “Nexon-profile”, we consider the player a potential Nexon player.  
  
The Nexon-profile is composed of features such as total time played by genre, % of time spent playing MMORPG, or % of purchase that’s MMORPG, ...etc.  

Finally, we use a K-nearest Neighbor model to do the profile matching and identify potential Nexon players.

## Model
See __potential\_player.ipynb__