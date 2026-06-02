from app import App


if __name__ == "__main__":
    config_path = "configs/vino_2st.json"

    app = App(config_path)
    app.main_loop()
    app.shutdown()
