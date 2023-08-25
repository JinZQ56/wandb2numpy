from wandb2numpy.config_loader import load_config
from wandb2numpy.export import export_data
from wandb2numpy.save_experiment import create_output_dirs, save_matrix

if __name__ == "__main__":
    default_config = "/home/lige/Codes/wandb2numpy/example_configs/metaworld_tcp.yaml"
    env_name_list = ["AssemblyProDMPTCP-v2",
                     "PickOutOfHoleProDMPTCP-v2",
                     "PlateSlideProDMPTCP-v2",
                     "PlateSlideBackProDMPTCP-v2",
                     "PlateSlideSideProDMPTCP-v2",
                     "PlateSlideBackSideProDMPTCP-v2",
                     "BinPickingProDMPTCP-v2",
                     "HammerProDMPTCP-v2",
                     "SweepIntoProDMPTCP-v2",
                     "BoxCloseProDMPTCP-v2",
                     "ButtonPressProDMPTCP-v2",
                     "ButtonPressWallProDMPTCP-v2",
                     "ButtonPressTopdownProDMPTCP-v2",
                     "ButtonPressTopdownWallProDMPTCP-v2",
                     "CoffeeButtonProDMPTCP-v2",
                     "CoffeePullProDMPTCP-v2",
                     "CoffeePushProDMPTCP-v2",
                     "DialTurnProDMPTCP-v2",
                     "DisassembleProDMPTCP-v2",
                     "DoorCloseProDMPTCP-v2",
                     "DoorLockProDMPTCP-v2",
                     "DoorOpenProDMPTCP-v2",
                     "DoorUnlockProDMPTCP-v2",
                     "HandInsertProDMPTCP-v2",
                     "DrawerCloseProDMPTCP-v2",
                     "DrawerOpenProDMPTCP-v2",
                     "FaucetOpenProDMPTCP-v2",
                     "FaucetCloseProDMPTCP-v2",
                     "HandlePressSideProDMPTCP-v2",
                     "HandlePressProDMPTCP-v2",
                     "HandlePullSideProDMPTCP-v2",
                     "HandlePullProDMPTCP-v2",
                     "LeverPullProDMPTCP-v2",
                     "PegInsertSideProDMPTCP-v2",
                     "PickPlaceWallProDMPTCP-v2",
                     "ReachProDMPTCP-v2",
                     "PushBackProDMPTCP-v2",
                     "PushProDMPTCP-v2",
                     "PickPlaceProDMPTCP-v2",
                     "PegUnplugSideProDMPTCP-v2",
                     "SoccerProDMPTCP-v2",
                     "StickPushProDMPTCP-v2",
                     "StickPullProDMPTCP-v2",
                     "PushWallProDMPTCP-v2",
                     "ReachWallProDMPTCP-v2",
                     "ShelfPlaceProDMPTCP-v2",
                     "SweepProDMPTCP-v2",
                     "WindowOpenProDMPTCP-v2",
                     "WindowCloseProDMPTCP-v2",
                     "BasketballProDMPTCP-v2"]

    list_doc = load_config(default_config)
    root_output_path = list_doc['experiment1']['output_path']
    for env_name in env_name_list:
        list_doc['experiment1']['config']['sampler.args.env_id']['values'][
            0] = env_name
        list_doc['experiment1'][
            'output_path'] = root_output_path + "/" + env_name
        experiment_data_dict, config_list = export_data(list_doc)
        for i, experiment in enumerate(experiment_data_dict.keys()):
            experiment_dir = create_output_dirs(config_list[i], experiment)
            print(experiment_dir)

            for field in experiment_data_dict[experiment]:
                save_matrix(experiment_data_dict[experiment], experiment_dir,
                            field, True, config_list[i])
