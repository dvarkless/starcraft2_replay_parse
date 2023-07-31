from replay_tools import ReplayTransformer

replay_dir = "./replays_sample"
replay_transformer = ReplayTransformer(50000, 1000, "./game_info.csv")
replay_transformer.parse_replays(replay_dir)
