import json
import io
import pickle
import logging
import traceback
import sklearn
import sys
import pathlib
import os
import pandas as pd
from git import Repo
from shutil import copyfile
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    model = None
    repo = None
    target_file = None
    app_path = str(pathlib.Path(__file__).parent.absolute())
    #repo_path = r"/Users/davidlaredorazo/Documents/Projects/Rappi Challenge/models_and_data"
    #app_path = r"/usr/src/web_app"
    repo_path = os.path.join(app_path, "../models_and_data")
    model_path = ''
    data_path = ''
    X_train = None
    y_train = None

    print(app_path)
    print(repo_path)

    #Configure logger
    training_logger = logging.getLogger('training_logger')
    training_logger.setLevel(logging.INFO)
    training_fh = logging.FileHandler('training.log')
    training_formatter = logging.Formatter(fmt='%(levelname)s:%(threadName)s:%(asctime)s:%(filename)s:%(funcName)s:%(message)s',
                                    datefmt='%m/%d/%Y %H:%M:%S')
    training_fh.setFormatter(training_formatter)
    training_logger.addHandler(training_fh)

    #Open config file
    try:
        with open(app_path + '/config.json') as fp:
            data = json.load(fp)
    except Exception as e:
        training_logger.error('Could not open config file')
        training_logger.error(traceback.format_exc())
        print('Could not open config file. Please check log')
        sys.exit(-1)

    #Try to open repository
    try:
        repo = Repo(repo_path)
    except Exception as e:
        training_logger.error('Could not open repository')
        training_logger.error(traceback.format_exc())
        print('Could not open repository. Please check log')
        sys.exit(-1)

    #Load model
    try:

        model_path = 'models' + '/' + data['model_label'] + '.pkl'

        if data['model_version']:
            commit = repo.commit(data['model_version'])
            target_file = commit.tree / model_path

            with io.BytesIO(target_file.data_stream.read()) as f:
                model = pickle.load(f)
        else:
            model = pickle.load(open(repo_path + '/' + model_path, 'rb'))
            data['model_version'] = repo.commit()

    except Exception as e:
        training_logger.error('Could not open file of the model')
        training_logger.error(traceback.format_exc())
        print('Could not open file of the model. Please check log')
        sys.exit(-1)


    #Load data
    try:

        data_path = './data/train.csv'

        if data['data_version']:
            commit = repo.commit(data['data_version'])
            target_file = commit.tree / data_path

            with io.BytesIO(target_file.data_stream.read()) as f:
                titanic_data = pd.read_csv(data_path)
        else:
            titanic_data = pd.read_csv(repo_path + '/' + data_path)
            data['data_version'] = 'latest'

        X_all = titanic_data.drop('Survived', axis=1)
        y_all = titanic_data.Survived

        X_train, y_train = X_all, y_all

        print(X_train)

    except Exception as e:
        training_logger.error('Could not load data')
        training_logger.error(traceback.format_exc())
        print('Could not load data. Please check log')
        sys.exit(-1)


    #Train model
    try:
        if data['training_type'] == 'new': #Reset model
            model = sklearn.base.clone(model)

        model.fit(X_train, y_train)
        training_logger.info('Model ' + data['model_label'] + '/' + str(data['model_version']) + ' with data version: ' + str(data['data_version']) +' successfully trained')
        training_logger.info('Training accuracy: ' + str(model.score(X_train, y_train)))
        print('Model ' + data['model_label'] + '/' + str(data['model_version']) + ' with data version: ' + str(data['data_version']) +' successfully trained')

    except Exception as e:
        training_logger.error('Could not train model')
        training_logger.error(traceback.format_exc())
        print('Could not train model. Please check log')
        sys.exit(-1)


    #Push model to the repository
    try:
        pickle.dump(model, open(repo_path + '/' + model_path, 'wb'))
        copyfile(repo_path + '/' + model_path, repo_path + '/' + 'models/deploy/' + data['model_label'] + '.pkl')
        repo.git.add(model_path)
        repo.git.add('models/deploy/' + data['model_label'] + '.pkl')
        repo.index.commit('Model ' + data['model_label'] + '/' + str(data['model_version']) + ' with data version: ' + str(data['data_version']))
        origin = repo.remote(name='origin')
        origin.push()
        training_logger.info('Model uploaded to git')
        print('Model uploaded to git')
    except Exception as e:
        #Need to reset git to previous state
        repo.git.reset('--hard')
        training_logger.error('Could not update git repository')
        training_logger.error(traceback.format_exc())
        print('Could not update git repository. Please check log')
        sys.exit(-1)

    sys.exit(0)


