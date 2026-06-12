from app import App
from utils.profiler import Profiler


if __name__ == "__main__":
    config_path = "configs/vino_1st_q.json"

    app = App(config_path)
    app.main_loop()
    app.shutdown()

    Profiler.summary()
