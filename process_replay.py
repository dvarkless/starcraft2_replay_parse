from replay_tools import BuildOrderData
from replay_tools import ReplayData
from pathlib import Path

replay_dir = Path("./replays_sample")
replay_transformer = BuildOrderData(30*60*16, 16*2, "./game_info.csv")
replay = "./replays_sample/all_upgrades_protoss.SC2Replay"
replay_data = ReplayData().parse_replay(replay).as_dict()

for player_dict in replay_transformer.yield_unit_counts(replay_data):
    print(player_dict)
