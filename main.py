import comet_ml
from data_loader.perfect_store_data_loader import PerfectStoreDataLoader
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
    data_loader = PerfectStoreDataLoader(config)

    # import time
    # t = []
    # i = 0
    # start = time.time()
    # for images, label in data_loader.get_train_generator():
    #     t.append(time.time() - start)
    #     print(t[-1])
    #     i = i + 1
    #     if i > 10:
    #         break
    #     start = time.time()
    # print('total: {}'.format(sum(t)))
    # print('avg: {}'.format(sum(t)/len(t)))
    #
    # return

    if phase == 'train':
        print('Create the model.')
        model = TripletLossModel(config)

        print('Create the trainer')
        trainer = TripletLossModelTrainer(model.model, data_loader, config)

        print('Start training the model.')
        trainer.train()



if __name__ == '__main__':
    main()
