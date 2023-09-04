<div align="center">

# Starcraft II replay parsing tool  

This library converts Starcraft2 replays into a sparse unit count table.

[Setup](#setup) •
[Usage](#usage) •
</div>

## About the Library
This library helps converting replay data into a format usable in data analysis 
or in training reinforced ML bots.  
It parses replay into a series of events and then makes a table in which you can 
find a count of every unit, building or resource of each player in each tick of 
the game.  
*Available functionality*:  

- Extract general replay data such as map name, players nicknames using ReplayData class.  
- Create a dict of lists representing player's build order in the span of the game.  

*Limitations to consider*:  

- The only available game mode is **1v1**.  
- Made for game version `5.0.11`

### Prerequisites
- `Python <= 3.9` (the latest sc2replay library is available in Python version 3.9)   
- Packages listed in `requirements.txt`

### Setup
1. Clone the repository by running

```sh
git clone https://github.com/dvarkless/starcraft2_replay_parse.git
```    
2. Create a python virtual environment:

```sh
cd Classic-ML-Models
python -m venv venv
```   
3. If you are using Linux or Mac:

```sh
source ./venv/bin/activate
```  
If you are using Windows:

```sh
./venv/Scripts/activate.ps1
```  
4. Install packages:

```sh
pip install -r requirements.txt
```

## Usage
1. Prepare a dataset, split it into training data, evaluation input and evaluation answers:  
The example code to run sample replays are provided in `process_replay.py` file.  
Here is the minimum code to use the library by the intended way. 

```python
from replay_tools import BuildOrderData, ReplayData

replay = 'replay_name.SC2Replay'
replay_data = ReplayData().parse_replay(path).as_dict()
print(replay_data['map_name'])
```

```
>>> Gresvan
```

```python
GAME_INFO_PATH = "./data/game_info.csv"
MAX_REPLAY_LEN = 30 * 60 * 16 # 30 minutes
STEP_LEN = 16 * 2 # write game state every ingame 2 seconds

replay_transformer = BuildOrderData(MAX_REPLAY_LEN, STEP_LEN, GAME_INFO_PATH)
for player_dict in replay_transformer.yield_unit_counts(replay_data):
	print(player_dict)  # There is two players in this game, 
						# so the method returns two dicts
```

```python
>>> {
>>>		'Zergling': [0, 0, 0, 0, 0 ....],
>>>		'Drone': [12, 12, 13, 13, 13 ....],
>>>		...
>>> }
>>> ...
```

## Output data structure:
#### ReplayData class
`out = ReplayData.as_dict()`  
`print(out)`

```python
out = {
    "processed_on": datetime.timestamp,
    "replay_name": str,
    "expansion": str,                       # ['WoL', 'HotS', 'Lotv']
    "frames": int,                          # Number of ticks the game has
    "mode": str,			    # '1v1'
    "map": str,				    # Hash value of the map
    "map_name": str,			    # Map name (prefix and suffix excluded)
    "matchup": str,			    # ZvT, ZvP, etc...
    "winners": List[str],		    # Nickname of the winner
    "losers": List[str],		    # Nickname of the loser(s)
    "stats_names": str,			    # Players_data dick keys
    "players": str,			    # Player nicknames
    "players_hash": str,		    # Hash of two players nicknames,
                                            # helps find identical replays 
					    # with different names
    "players_data": dict{		    # Players info
				'id': int,
				'full_name':str,	# name as in stats_names
				'race': str,
				'league': int,		# 0-8, 0-unranked, 8-GM		
				'url': str,		# link to battle.net account
				'is_winner': bool,	
						},					
    "stats": dict { ... },	            # Events
    "league": int,			    # Min players league: 0-8, 0-unranked, 8-GM	
}

```
#### BuildOrderData class
`out = BuildOrderData.yield_unit_counts(replay_data)`  
`print(out)`

```python
# out = Generator[dict[...]]
player_1_dict = next(out)
```
The dict represents a sparse table where columns are defined in the `game_info.csv`
file. There is units and buildings, regardless of players' game race.
Additionally, there is a minerals and vespene counter.  
The lists are the same length of either the game length in `tick // parsing_step` 
or maximum game length, depending on which one is smaller.  

**Columns**:  

- Units  
- Buildings  
- Upgrades  
- Resources available  

**Rows**:  

- Each position represents an entity count of the current type in the current tick.  
- Each list in the dict has the same length.  
- Type: `int` > 0

```python
player_1_dict = {
	terran_unit: list[int],
	...,
	zerg_upgrade: list[int],
	...,
	protoss_building: list[int],
	...,
	minerals_available: list[int],
	vespene_available: list[int],
}
```
## Acknowledgements
This project uses [SC2Reader](https://github.com/ggtracker/sc2reader) tool to parse replays

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
