# [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

A classic text classification task,  please download the .csv files and put them into a ./data folder in your local enviroment. 

## Generate Config     
```
himl hiera/model=textcnn/ --output-file config/config.yaml
```

## Run 
```
python run.py --config config/config.yaml 
```
