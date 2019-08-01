import comet_ml
from data_loader.data_loader import DataLoader
from models.triplet_loss_model import TripletLossModel
from models.knn_model import KNNModel
from trainers.triplet_loss_trainer import TripletLossModelTrainer
from evaluater.evaluater import ModelEvaluater
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        phase = args.phase
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

    if phase == 'train':
        print('Create the model.')
        model = TripletLossModel(config)

        print('Create the trainer')
        trainer = TripletLossModelTrainer(model.model, data_loader, config)

        print('Start training the model.')
        trainer.train()

    else:
        print('Create the model.')
        embedding_model = TripletLossModel(config)
        knn_model = KNNModel(config, embedding_model)

        print('Create the evaluater')
        evaluater = ModelEvaluater(knn_model, data_loader, config)

        print('Start evaluate the model.')
        evaluater.evaluate()



if __name__ == '__main__':
    main()
