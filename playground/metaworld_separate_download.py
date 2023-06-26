
from wandb2numpy.config_loader import load_config
from wandb2numpy.export import export_data
from wandb2numpy.save_experiment import create_output_dirs, save_matrix


if __name__=="__main__":
    default_config = "/home/hongyi/Codes/alr_ma/wandb2numpy/example_configs/metaworld.yaml"
    env_name_list = [ "AssemblyProDMP-v2",
            "PickOutOfHoleProDMP-v2",
            "PlateSlideProDMP-v2",
            "PlateSlideBackProDMP-v2",
            "PlateSlideSideProDMP-v2",
            "PlateSlideBackSideProDMP-v2",
            "BinPickingProDMP-v2",
            "HammerProDMP-v2",
            "SweepIntoProDMP-v2",
            "BoxCloseProDMP-v2",
            "ButtonPressProDMP-v2",
            "ButtonPressWallProDMP-v2",
            "ButtonPressTopdownProDMP-v2",
            "ButtonPressTopdownWallProDMP-v2",
            "CoffeeButtonProDMP-v2",
            "CoffeePullProDMP-v2",
            "CoffeePushProDMP-v2",
            "DialTurnProDMP-v2",
            "DisassembleProDMP-v2",
            "DoorCloseProDMP-v2",
            "DoorLockProDMP-v2",
            "DoorOpenProDMP-v2",
            "DoorUnlockProDMP-v2",
            "HandInsertProDMP-v2",
            "DrawerCloseProDMP-v2",
            "DrawerOpenProDMP-v2",
            "FaucetOpenProDMP-v2",
            "FaucetCloseProDMP-v2",
            "HandlePressSideProDMP-v2",
            "HandlePressProDMP-v2",
            "HandlePullSideProDMP-v2",
            "HandlePullProDMP-v2",
            "LeverPullProDMP-v2",
            "PegInsertSideProDMP-v2",
            "PickPlaceWallProDMP-v2",
            "ReachProDMP-v2",
            "PushBackProDMP-v2",
            "PushProDMP-v2",
            "PickPlaceProDMP-v2",
            "PegUnplugSideProDMP-v2",
            "SoccerProDMP-v2",
            "StickPushProDMP-v2",
            "StickPullProDMP-v2",
            "PushWallProDMP-v2",
            "ReachWallProDMP-v2",
            "ShelfPlaceProDMP-v2",
            "SweepProDMP-v2",
            "WindowOpenProDMP-v2",
            "WindowCloseProDMP-v2",
            "BasketballProDMP-v2"]

    list_doc = load_config(default_config)
    root_output_path = list_doc['experiment1']['output_path']
    for env_name in env_name_list:
        list_doc['experiment1']['config']['sampler.args.env_id']['values'][0] = env_name
        list_doc['experiment1']['output_path'] = root_output_path + "/" + env_name
        experiment_data_dict, config_list = export_data(list_doc)
        for i, experiment in enumerate(experiment_data_dict.keys()):
            experiment_dir = create_output_dirs(config_list[i], experiment)
            print(experiment_dir)

            for field in experiment_data_dict[experiment]:
                save_matrix(experiment_data_dict[experiment], experiment_dir, field, True, config_list[i])