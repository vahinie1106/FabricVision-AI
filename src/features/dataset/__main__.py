from .dataset_manager import DatasetManagementConfig, DatasetManager


def main() -> None:
    config = DatasetManagementConfig()
    manager = DatasetManager(config=config)
    manager.run()


if __name__ == "__main__":
    main()
