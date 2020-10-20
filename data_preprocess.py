import pandas as pd
import logging
import traceback
import os
import sys
import pathlib
from git import Repo

if __name__ == '__main__':

    app_path = str(pathlib.Path(__file__).parent.absolute())
    repo_path = os.path.join(app_path, "../models_and_data")
    #repo_path = r"/Users/davidlaredorazo/Documents/Projects/Rappi Challenge/models_and_data"
    #app_path = r"/usr/src/web_app"
    #repo_path = app_path + "/models_and_data"
    file_exists = False

    #Configure logger
    data_logger = logging.getLogger('data_logger')
    data_logger.setLevel(logging.INFO)
    data_fh = logging.FileHandler('./data.log')
    data_formatter = logging.Formatter(fmt='%(levelname)s:%(threadName)s:%(asctime)s:%(filename)s:%(funcName)s:%(message)s',
                                    datefmt='%m/%d/%Y %H:%M:%S')
    data_fh.setFormatter(data_formatter)
    data_logger.addHandler(data_fh)


    #Try to open repository
    try:
        repo = Repo(repo_path)
    except Exception as e:
        data_logger.error('Could not open repository')
        data_logger.error(traceback.format_exc())
        print('Could not open repository. Please check log')
        sys.exit(-1)


    #Pre-process data according to solution
    try:

        training = pd.read_csv(repo_path + "/data_raw/train.csv")
        training = training.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

        #Fill missing values
        training.Age = training.Age.fillna(training.Age.median())
        training.Embarked = training.Embarked.fillna('S')

        #Transform categorical into integer
        embark_dummies_titanic = pd.get_dummies(training['Embarked'])
        sex_dummies_titanic = pd.get_dummies(training['Sex'])
        pclass_dummies_titanic = pd.get_dummies(training['Pclass'], prefix="Class")

        #Put data together
        training = training.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
        titanic = training.join([embark_dummies_titanic, sex_dummies_titanic, pclass_dummies_titanic])

        print(titanic)

        data_logger.info('Successfully pre processed data')
        print("Successfully pre processed data")

    except Exception as e:

        print("Error while processing data. Please check log")
        data_logger.error("Error while processing data")
        data_logger.error(traceback.format_exc())
        sys.exit(-1)


    #Upload to git
    try:

        file_exists = os.path.isfile(repo_path + '/data/train.csv')
        titanic.to_csv(repo_path + '/data/train.csv', index=False)
        t = repo.head.commit.tree

        if repo.git.diff(t) or file_exists is False:

            repo.git.add(repo_path + '/data/train.csv')
            repo.index.commit('Adding pre-processed data')
            origin = repo.remote(name='origin')
            origin.push()
            data_logger.info('Data uploaded to git')
            print('Data uploaded to git')

        else:

            print("No changes detected in data.")
            data_logger.error('No changes detected in data.')

    except Exception as e:
        #Need to reset git to previous state
        repo.git.reset('--hard')
        data_logger.error('Could not update git repository.')
        data_logger.error(traceback.format_exc())
        print('Could not update git repository. Please check log.')
        sys.exit(-1)

    sys.exit(0)