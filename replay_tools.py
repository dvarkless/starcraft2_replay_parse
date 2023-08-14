from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import sc2reader
from sc2reader.constants import LOCALIZED_RACES
from sc2reader.engine.plugins import (APMTracker, ContextLoader,
                                      SelectionTracker)

from game_info import handlers

LOCALIZED_RACES["Терраны"] = "Terran"
LOCALIZED_RACES["Зерги"] = "Zerg"
LOCALIZED_RACES["Протоссы"] = "Protoss"


class ReplayData:
    __parsers__ = handlers

    def __init__(self):
        self.players = defaultdict(lambda: defaultdict(list))
        self.replay = None
        self.winners = []
        self.losers = []
        self.expansion = None

    def parse_replay(self, replay=None):
        # This is the engine that holds some required plugins for parsing
        engine = sc2reader.engine.GameEngine(
            plugins=[ContextLoader(), APMTracker(), SelectionTracker()]
        )

        if isinstance(replay, sc2reader.resources.Replay):
            pass
        elif isinstance(replay, (Path, str)):
            replay = str(replay)
            replay = sc2reader.load_replay(replay, engine=engine)
        else:
            raise ValueError(f"Unknown type for variable replay: [{type(replay)}]")

        self.players = defaultdict(lambda: defaultdict(list))
        self.replay = replay
        # Get the number of frames (one frame is 1/16 of a second)
        self.frames = replay.frames
        # Gets the game mode (if available)
        self.game_mode = replay.real_type
        # Gets the map hash (if we want to download the map, or do map-based analysis)
        self.map_hash = replay.map_hash
        map_name = replay.map_name.removeprefix("[ESL]")
        map_name = map_name.removesuffix("LE")
        self.map_name = map_name.strip()

        leagues = []
        for player in replay.player.values():
            try:
                league = player.highest_league
            except AttributeError:
                league = 0
            leagues.append(league)
        leagues = [num for num in leagues if num > 0]
        self.league = min(leagues if leagues else [0])

        # Use the parsers to get data
        for event in replay.events:
            for parser in self.__parsers__:
                parser(self, event)

        map_names = {v: v.detail_data["name"] for v in self.replay.player.values()}

        # Check if there was a winner
        if replay.winner is not None:
            self.winners = replay.winner.players
            self.losers = [p for p in replay.players if p not in replay.winner.players]
        else:
            self.winners = []
            self.losers = [p for p in replay.players]

        self.winners = [map_names[p] for p in self.winners]
        self.losers = [map_names[p] for p in self.losers]
        # Check to see if expansion data is available
        self.expansion = replay.expansion
        self.players_hash = replay.people_hash
        self.is_ranked = bool(replay.is_ladder)
        self.player_names = self.winners + self.losers
        self.players_data = {k: dict() for k in self.player_names}

        for player_data in replay.players:
            name = map_names[player_data]
            self.players_data[name]["id"] = player_data.detail_data["bnet"]["uid"]
            self.players_data[name]["full_name"] = str(player_data)
            self.players_data[name]["race"] = LOCALIZED_RACES[player_data.play_race]
            self.players_data[name]["league"] = getattr(
                player_data, "highest_league", 0
            )
            self.players_data[name]["url"] = getattr(player_data, "url", "")

            self.players_data[name]["is_winner"] = name in self.winners

        return self

    def as_dict(self):
        return {
            "processed_on": datetime.utcnow().isoformat(),
            "replay_name": self.replay,
            "expansion": self.expansion,
            "frames": self.frames,
            "mode": self.game_mode,
            "map": self.map_hash,
            "map_name": self.map_name,
            "matchup": "v".join(
                sorted(
                    [
                        self.players_data[s]["race"][0].upper()
                        for s in self.winners + self.losers
                    ]
                )
            ),
            "winners": [s for s in self.winners],
            "losers": [s for s in self.losers],
            "stats_names": [k for k in self.players[1].keys()],
            "players": self.player_names,
            "players_hash": self.players_hash,
            "players_data": self.players_data,
            "stats": {player: data for player, data in self.players.items()},
            "league": self.league,
        }


class BuildOrderData:
    general_data = [
        "worker_event",
        "army_event",
        "ground_building",
        "air_building",
        "tech_building",
        "expansion_event",
        "vespene_event",
    ]
    upgrades_data = ["upgrades"]

    specific_data = [
        "minerals_available",
        "vespene_available",
    ]

    symbol_meaning = {
        "+": "create_",
        "-": "lose_",
        "*": "morth_",
    }
    buildings_at_start = [
        "Hatchery",
        "Nexus",
        "CommandCenter",
    ]

    replace_units = {
        "SiegeTankSieged": "SiegeTank",
    }

    morthed_units = {
        "SiegeTankSieged": "SiegeTank",
        "GreaterSpire": "Spire",
        "OrbitalCommand": "CommandCenter",
        "PlanetaryFortress": "CommandCenter",
        "WarpGate": "Gateway",
        "Archon": "HighTemplar",
        "Lair": "Hatchery",
        "Hive": "Hatchery",
        "Lurker": "Hydralisk",
        "Baneling": "Zergling",
        "Ravager": "Roach",
        "BroodLord": "Corruptor",
        "Overseer": "Overlord",
    }
    ignore_morths = [
        "SiegeTank",
        "SiegeTankSieged",
        "Hellbat",
        "Hellion",
        "Liberator",
        "LiberatorAG",
        "InfestorBurrowed",
        "RoachBurrowed",
        "ZerglingBurrowed",
        "BanelingBurrowed",
        "DroneBurrowed",
        "QueenBurrowed",
        "HydraliskBurrowed",
        "LurkerBurrowed",
        "SwarmHostBurrowed",
        "UltraliskBurrowed",
        "WarpGate",
        "WarpPrism",
        "WarpPrismPhasing",
        "Observer",
        "Overseer",
    ]
    game_speed = {
        "normal": 0.6,
        "fast": 0.8,
        "faster": 1.0,
    }
    ticks_per_second = 16

    def __init__(
        self,
        max_tick: int,
        bin_size_ticks: int,
        game_data,
        add_info=None,
        game_data_speed="faster",
    ) -> None:
        self.tick_bins = [
            t * bin_size_ticks for t in range(max_tick // bin_size_ticks + 1)
        ]
        self.max_tick = self.tick_bins[-1]
        self.game_max_dur = self.max_tick
        self.bin_size_ticks = bin_size_ticks
        self.bins_len = len(self.tick_bins)

        self.add_info = add_info
        if add_info is None:
            self.add_info = dict()
        else:
            for k in self.add_info.keys():
                if not isinstance(k, str):
                    raise ValueError(
                        "Keys of additional info dict should only be of type str"
                    )

        self.game_data = self.get_game_data(game_data, game_data_speed)

    def get_game_data(self, path, speed="faster"):
        df = pd.read_csv(path, index_col="name")

        for name in df.index:
            if name in self.replace_units.keys():
                df = df.drop(index=name)
                continue
            if df.loc[name, "type"] in ("Unit", "Building"):
                prefix_list = list(self.symbol_meaning.values())
                if name in self.morthed_units.keys():
                    prefix_list.remove("create_")
                else:
                    prefix_list.remove("morth_")

                for prefix in prefix_list:
                    new_name = prefix + name
                    df.loc[new_name, :] = df.loc[name, :]
                    build_ticks = int(
                        df.loc[new_name, "build_time"]
                        * self.game_speed[speed]
                        * self.ticks_per_second
                    )
                    df.loc[new_name, "build_time"] = build_ticks
                    if prefix == "lose_":
                        df.loc[new_name, "build_time"] = 0
                df = df.drop(index=name)

        return df

    def get_event_counts(self, player_data_dict):
        events_list = []
        to_iter = self.general_data + self.upgrades_data
        for name in to_iter:
            for event in player_data_dict[name]:
                events_list.append(self.normalize_event(event))
        for name in self.specific_data:
            for event in player_data_dict[name]:
                events_list.append(self.normalize_event(event, event_name=name))
        transformed_events = [self.transform_event(*e) for e in events_list]
        transformed_events = [e for e in transformed_events if e is not None]
        return transformed_events

    def get_game_duration(self, replay_data_dict):
        return min(int(replay_data_dict["frames"]), self.max_tick)

    def get_ticks(self):
        return [
            self.bin_size_ticks * i
            for i in range(self.game_max_dur // self.bin_size_ticks + 1)
        ]

    def yield_unit_counts(self, replay_data_dict):
        self.game_max_dur = self.get_game_duration(replay_data_dict)
        for player_events in replay_data_dict["stats"].values():
            transformed_events = self.get_event_counts(player_events)
            specific_events = []
            regular_events = []
            for event in transformed_events:
                if event[1] in self.specific_data:
                    specific_events.append(event)
                else:
                    regular_events.append(event)

            dense_dict = self.get_density_from_events(regular_events)
            build_order_dict = self.get_build_order_from_density(dense_dict)

            build_order_special = self.get_build_order_special(specific_events)
            build_order_dict |= build_order_special
            yield build_order_dict

    def init_zeros_density(self, specific=False):
        data_dict = dict()
        if not specific:
            for key in self.game_data.index:
                data_dict[key] = [
                    0 for _ in range(self.game_max_dur // self.bin_size_ticks + 1)
                ]
                if key in self.buildings_at_start:
                    data_dict[key][0] += 1
        else:
            for key in self.specific_data:
                data_dict[key] = [
                    0 for _ in range(self.game_max_dur // self.bin_size_ticks + 1)
                ]

        return data_dict

    def get_density_from_events(self, event_list):
        density_dict = self.init_zeros_density()
        for time_pos, event_name, event_value in event_list:
            density_dict[event_name][time_pos // self.bin_size_ticks] += event_value

        return density_dict

    def get_build_order_special(self, event_list):
        density_dict = self.init_zeros_density(specific=True)

        for time_pos, event_name, event_value in event_list:
            density_dict[event_name][time_pos // self.bin_size_ticks] += event_value

        for name, vals in density_dict.items():
            curr_val = 0
            for i, val in enumerate(vals):
                curr_val = curr_val if val == 0 else val
                density_dict[name][i] = curr_val

        return density_dict

    def get_build_order_from_density(self, density_dict):
        build_order_dict = dict()
        for event_name, dense_vals in density_dict.items():
            relevant_prefixes = ("create", "lose", "morth")
            if any([event_name.startswith(p) for p in relevant_prefixes]):
                action, name = event_name.split("_")
            else:
                action = "create"
                name = event_name
            build_order_dict[name] = build_order_dict.get(
                name, [0 for _ in range(self.game_max_dur // self.bin_size_ticks + 1)]
            )

            sign = 1
            if action == "lose":
                sign = -1

            curr_val = 0
            for i, val in enumerate(dense_vals):
                curr_val += val * sign
                build_order_dict[name][i] += max(curr_val, 0)
        return build_order_dict

    def normalize_event(self, event_data, event_name=""):
        tick = 0
        event_value = 1
        action = ""
        if len(event_data) <= 2:
            tick, event_obj = event_data
        else:
            tick, action, event_obj = event_data

        try:
            int(event_obj)
        except ValueError:
            event_name = event_obj
        else:
            event_value = event_obj

        return tick, action, event_name, event_value

    def transform_event(self, tick, action, event_name, event_value):
        if tick >= self.game_max_dur:
            return
        if action == "*" and event_name in self.ignore_morths:
            return
        if action == "*" and event_name not in self.morthed_units.keys():
            return
        if event_name in self.replace_units.keys():
            event_name = self.replace_units[event_name]

        if action == "+" and event_name in self.morthed_units.keys():
            event_name = self.symbol_meaning[action] + self.morthed_units[event_name]
        elif action in ("+", "*", "-"):
            event_name = self.symbol_meaning[action] + event_name

        if action in ("+", "*"):
            creation_tick = tick - self.game_data.loc[event_name, "build_time"]
        else:
            creation_tick = tick

        creation_tick = max(creation_tick, 0)
        return (int(creation_tick), event_name, event_value)
