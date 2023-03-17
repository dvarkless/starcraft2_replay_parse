from collections import defaultdict
from datetime import datetime
from pathlib import Path
from traceback import print_exc

import numpy as np
import pandas as pd
import sc2reader
import xarray as xr
from sc2reader.engine.plugins import (APMTracker, ContextLoader,
                                      SelectionTracker)

from game_info import handlers


class ReplayData:
    __parsers__ = handlers

    @classmethod
    def parse_replay(cls, replay=None, replay_file=None, file_object=None):

        replay_data = ReplayData(replay_file)
        try:
            # This is the engine that holds some required plugins for parsing
            engine = sc2reader.engine.GameEngine(
                plugins=[ContextLoader(), APMTracker(), SelectionTracker()])

            if replay:
                pass
            elif replay_file and not file_object:
                # Then we are not using ObjectStorage for accessing replay files
                replay = sc2reader.load_replay(replay_file, engine=engine)
            elif file_object:
                # We are using ObjectStorage to access replay files
                replay = sc2reader.load_replay(file_object, engine=engine)
            else:
                pass
            # Get the number of frames (one frame is 1/16 of a second)
            replay_data.frames = replay.frames
            # Gets the game mode (if available)
            replay_data.game_mode = replay.real_type
            # Gets the map hash (if we want to download the map, or do map-based analysis)
            replay_data.map_hash = replay.map_hash
            map_name = replay.map_name.removeprefix('[ESL]')
            map_name = map_name.removesuffix('LE')
            replay_data.map_name = map_name.strip()

            # Use the parsers to get data
            for event in replay.events:
                for parser in cls.__parsers__:
                    parser(replay_data, event)

            # Check if there was a winner
            if replay.winner is not None:
                replay_data.winners = replay.winner.players
                replay_data.losers = [
                    p for p in replay.players if p not in replay.winner.players]
            else:
                replay_data.winners = []
                replay_data.losers = [p for p in replay.players]
            # Check to see if expansion data is available
            replay_data.expansion = replay.expansion
            return replay_data
        except:
            # print our error and return NoneType object
            print_exc()
            return None

    def as_dict(self):
        return {
            "processed_on": datetime.utcnow().isoformat(),
            "replay_name": self.replay,
            "expansion": self.expansion,
            "frames": self.frames,
            "mode": self.game_mode,
            "map": self.map_hash,
            "map_name": self.map_name,
            "matchup": "v".join(sorted([s.detail_data["race"][0].upper() for s in self.winners + self.losers])),
            "winners": [(s.pid, s.name, s.detail_data['race']) for s in self.winners],
            "losers": [(s.pid, s.name, s.detail_data['race']) for s in self.losers],
            "stats_names": [k for k in self.players[1].keys()],
            "stats": {player: data for player, data in self.players.items()}
        }

    def __init__(self, replay):
        self.players = defaultdict(lambda: defaultdict(list))
        self.replay = replay
        self.winners = []
        self.losers = []
        self.expansion = None


class ReplayTransformer:
    columns = ['game_id', 'id', 'is_winner', 'race', 'matchup',
               'game_duration']

    general_data = ['army_event', 'ground_building', 'air_building',
                    'tech_building', 'expantion_event']
    upgrades_data = ['upgrades']

    specific_data = ['minerals_available', 'vespene_available',
                     'minerals_collection_rate', 'vespene_collection_rate']

    symbol_meaning = {'+': 'create_',
                      '-': 'lose_',
                      '*': 'morth_',
                      }

    morthed_units = {'GreaterSpire': 'Spire',
                     'OrbitalCommand': 'CommandCenter',
                     'PlanetaryFortress': 'CommandCenter',
                     'WarpGate': 'Gateway',
                     'Archon': 'HighTemplar',
                     'Lair': 'Harchery',
                     'Hive': 'Hatchery',
                     'Lurker': 'Hydralisk',
                     'Baneling': 'Zergling',
                     'Ravager': 'Roach',
                     'BroodLord': 'Corruptor',
                     'Overseer': 'Overlord',
                     }
    ignore_morths = [
        'SiegeTank', 'SiegeTankSieged', 'Hellbat', 'Hellion',
        'Liberator', 'LiberatorAG', 'InfestorBurrowed',
        'RoachBurrowed', 'ZerglingBurrowed', 'BanelingBurrowed',
        'DroneBurrowed', 'QueenBurrowed', 'HydraliskBurrowed',
        'LurkerBurrowed', 'SwarmHostBurrowed', 'UltraliskBurrowed',
        'WarpGate',
    ]
    game_speed = {
        'normal': 1,
        'fast': 1.2,
        'faster': 1.4,
    }
    ticks_per_second = 16

    def __init__(self, max_tick, bin_size_ticks,
                 game_data=None):
        self.tick_bins = [
            t*bin_size_ticks for t in range(max_tick//bin_size_ticks+1)]
        self.max_tick = self.tick_bins[-1]
        self.bin_size_ticks = bin_size_ticks
        self.bins_len = len(self.tick_bins)
        self.id = 0
        self.game_id = 0

        self.add_info = dict()

        self.game_data = self.get_game_data(game_data)

    def init_data_array(self):
        coords = dict()
        coords['time'] = np.array(self.tick_bins)
        coords['columns'] = list(self.game_data.index)
        for column in self.specific_data:
            coords['columns'].append(column)

        da = xr.DataArray(
            data=np.zeros((len(coords['time']), len(coords['columns']))),
            coords=coords,
            dims=["time", "columns"],

        )

        da.attrs = {k: 0 for k in self.columns}
        for k, v in self.add_info.items():
            da.attrs[k] = v
        return da

    def get_game_data(self, path: str):
        df = pd.read_csv(path, index_col='name')

        for name in df.index:
            if df.loc[name, 'type'] == 'Unit':
                prefix_list = list(self.symbol_meaning.values())
                if name in self.morthed_units.keys():
                    prefix_list.remove('create_')
                else:
                    prefix_list.remove('morth_')

                for prefix in prefix_list:
                    new_name = prefix + name
                    df.loc[new_name, :] = df.loc[name, :]
                    build_ticks = int(df.loc[new_name, 'build_time']
                                      * self.game_speed['faster']
                                      * self.ticks_per_second)
                    df.loc[new_name, 'build_time'] = build_ticks
                    if prefix == 'lose_':
                        df.loc[new_name, 'build_time'] = 0
                df = df.drop(index=name)

        return df

    def parse_replays(self, replay_dir, add_info=None):
        if add_info is not None:
            assert isinstance(add_info, dict)
            self.add_info = add_info.copy()

        print('Starting replay parse process')
        replay_dir = Path(replay_dir)
        processed_list = []
        dataset = xr.Dataset()

        for replay_path in replay_dir.iterdir():
            if replay_path.suffix == '.SC2Replay':
                print(replay_path)
                replay_dict = ReplayData.parse_replay(
                    replay_file=str(replay_path)).as_dict()
                packed_data = self.process_replay(replay_dict)
                for data in packed_data:
                    dataset[data.attrs['id']] = data
                processed_list.append(replay_path.stem)
                print(dataset)

        print('=== Replay parsing finished! ===')
        print(f'Number of replays parsed: {len(processed_list)}\n')
        print(f'With filenames: {processed_list}')

    def process_replay(self, replay_dict):
        output = []
        players_list = [(n, gr, 'L') for n, _, gr in replay_dict['losers']]
        players_list += [(n, gr, 'W') for n, _, gr in replay_dict['winners']]

        for player_N, player_game_race, outcome in players_list:
            self.da = self.init_data_array()

            self.da.attrs['game_id'] = self.game_id
            self.da.attrs['map_hash'] = replay_dict['map']
            self.da.attrs['map_name'] = replay_dict['map_name']
            self.da.attrs['matchup'] = replay_dict['matchup']
            self.da.attrs['is_winner'] = 1 if outcome == 'W' else 0
            self.da.attrs['race'] = player_game_race
            self.da.attrs['id'] = self.id
            self.da.attrs['game_duration'] = replay_dict['frames']
            self.supply_consumed_list = replay_dict['stats'][1]

            self.parse_player(replay_dict['stats'][player_N])
            self.id += 1
            output.append(self.da.copy())

        self.game_id += 1

        return output

    def parse_player(self, player_stats):
        for name in self.general_data:
            self.transform_event(player_stats[name])
        for name in self.upgrades_data:
            self.transform_event(player_stats[name], event_type='upgrade')
        for name in self.specific_data:
            self.transform_event(
                player_stats[name], event_type='specific', event_name=name)
        max_game_bin = self.da.attrs['game_duration']
        self.fill_all_invalid(self.da.attrs['race'], max_game_bin)

    def fill_all_invalid(self, game_race, game_max_bin):
        for entity in self.game_data.index:
            if self.game_data.loc[entity, 'race'] == game_race:
                self.da.loc[game_max_bin:, entity] = -1
            else:
                self.da.loc[:, entity] = -1

    def transform_event(self, event_list, event_type='general', event_name=''):
        tick = 0
        action = ''
        event_value = 1
        data_name = 'nop'

        for event_data in event_list:
            if event_type == 'general':
                tick, action, data_name = event_data
                if action == '+' and data_name in self.morthed_units.keys():
                    event_name = self.symbol_meaning[action] \
                        + self.morthed_units[data_name]
                else:
                    event_name = self.symbol_meaning[action] + data_name
            elif event_type == 'upgrade':
                tick, event_name = event_data
            elif event_type == 'specific':
                tick, event_value = event_data
            else:
                raise ValueError(f'"event type" should be in \
                                 ["general", "upgrades", "specific"]')

            if action == '*' and data_name in self.ignore_morths:
                continue
            if action == '*' and data_name not in self.morthed_units.keys():
                continue
            if event_name not in self.da.columns:
                msg = f'Bad key, "{event_name}" is not in columns'
                raise KeyError(msg)

            if tick < self.max_tick:
                if action == '*' and data_name in self.morthed_units.keys():
                    creation_tick = tick - \
                        self.game_data.loc[event_name, 'build_time']
                elif action == '+':
                    creation_tick = tick \
                        - self.game_data.loc[event_name, 'build_time']
                else:
                    creation_tick = tick

                time_pos = int(creation_tick - creation_tick %
                               self.bin_size_ticks)
                time_pos = max(time_pos, 0)
                self.da.loc[time_pos, event_name] += event_value
            else:
                continue

    def tick_supply_mapper(self, tick_val):
        return self.supply_consumed_list[tick_val//160][1]
