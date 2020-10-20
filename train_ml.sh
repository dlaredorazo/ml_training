#!/bin/bash
cd models_and_data
git pull
cd ..
cd web_app
git pull

KEY=$(jq .preprocess_data config.json)

if [ $KEY = '"True"' ]; then
  python data_preprocess.py
fi

if [ $? -eq 0 ]; then
  python train_model.py
else
  echo "Error while processing data. Check data logs"
fi

if [ $? -eq 0 ]; then
  echo "Model trained and uploaded"
else
  echo "Error while training model. Check app logs"
fi

