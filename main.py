import comet_ml
from data_loader.data_loader import DataLoader
from models.triplet_loss_model import TripletLossModel
from trainers.triplet_loss_trainer import TripletLossModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # args = get_args()
    # config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

    print('Create the data generator.')
    data_loader = DataLoader(config)

    print('Create the model.')
    model = TripletLossModel(config)

    print('Create the trainer')
    trainer = TripletLossModelTrainer(model.model, data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
