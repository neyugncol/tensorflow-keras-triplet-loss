import comet_ml
from data_loader.perfect_store_data_loader import PerfectStoreDataLoader
from data_loader.arcface_data_loader import ArcFaceDataLoader
from models.triplet_loss_model import TripletLossModel
from models.arcface_model import ArcFaceModel
from trainers.triplet_loss_trainer import TripletLossModelTrainer
from trainers.arcface_trainer import ArcFaceModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    #     args = get_args()
    #     phase = args.phase
    #     config = process_config(args.config)
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    args = get_args()
    phase = args.phase
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

    # print('Create the data generator.')
    # data_loader = PerfectStoreDataLoader(config)

    print('Create the data generator.')
    data_loader = ArcFaceDataLoader(config)

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
        # print('Create the model.')
        # model = TripletLossModel(config)
        #
        # print('Create the trainer')
        # trainer = TripletLossModelTrainer(model.model, data_loader, config)
        #
        # print('Start training the model.')
        # trainer.train()

        config.num_classes = len(data_loader.train_category_ids)

        print('Create the model.')
        model = ArcFaceModel(config)

        print('Create the trainer')
        trainer = ArcFaceModelTrainer(model.model, model.predict_model, data_loader, config)

        print('Start training the model.')
        trainer.train()

    # else:
    #     import numpy as np
    #     from scipy.spatial.distance import cdist
    #     from sklearn import metrics
    #     print('Load the model.')
    #     model = TripletLossModel(config)
    #     model.load(config.weight_file)
    #
    #     ########################################
    #     ref_images, ref_labels = data_loader.get_reference_data()
    #     ref_embeddings = model.predict(ref_images, batch_size=config.batch_size, verbose=1)
    #     eval_embeddings = []
    #     eval_labels = []
    #     for (images, labels) in data_loader.get_val_generator():
    #         embeddings = model.predict(images)
    #         eval_embeddings.append(embeddings)
    #         eval_labels.append(labels)
    #
    #     eval_embeddings = np.concatenate(eval_embeddings)
    #     eval_labels = np.concatenate(eval_labels)
    #     eval_categories = np.unique(eval_labels)
    #
    #     pairwise_distance = cdist(eval_embeddings, ref_embeddings, metric=config.distance_metric)
    #     predictions = ref_labels[np.argmin(pairwise_distance, axis=1)]
    #
    #     result = {
    #         'val_accuracy': metrics.accuracy_score(eval_labels, predictions),
    #         'val_precision': metrics.precision_score(eval_labels, predictions, labels=eval_categories, average='macro'),
    #         'val_recall': metrics.recall_score(eval_labels, predictions, labels=eval_categories, average='macro'),
    #         'val_f1_score': metrics.f1_score(eval_labels, predictions, labels=eval_categories, average='macro')
    #     }
    #
    #     print(' Result: {}'.format(' - '.join(['{}: {}'.format(key, value) for key, value in result.items()])))



if __name__ == '__main__':
    main()
