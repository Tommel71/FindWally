import pickle
import pandas as pd
from fastai.vision.all import *

class MultiModel:

    def __init__(self, path_list):
        self.path_list = path_list

    def load_model(self, path):
        # load pickled file
        with open(path, 'rb') as file:
            model = load_learner(file)
        return model

    def get_agreed_features_names(self, df):
        sum_features = df.sum(axis = 0) # TODO check
        agreed_features = sum_features.argsort()[-2:][::-1]
        agreed_features_names = list(df.columns[agreed_features])
        return agreed_features_names


    def pick_anomaly(self, df, agreed_features_names):
        sums = df[agreed_features_names].sum(axis = 1)# TODO check
        anomaly_index = sums.argmin()
        anomaly_name = df.index[anomaly_index]
        return anomaly_name

    def predict(self, list_of_imagebatches):

        imagebatch_dfs = self.list_of_imagebatches_to_feature_dfs(list_of_imagebatches)
        anomalies = []
        for df in imagebatch_dfs:
            agreed_features_names = self.get_agreed_features_names(df)
            anomaly = self.pick_anomaly(df, agreed_features_names)
            anomaly_clean = anomaly.split("/")[-1]
            anomalies.append(anomaly_clean)

        return anomalies

    def list_of_imagebatches_to_feature_dfs(self, list_of_imagebatches):

        imagebatch_dfs = []
        from collections import defaultdict
        dd = defaultdict(dict)
        for path in self.path_list:
            model = self.load_model(path)

            preds = dd[path]
            flattened = [im for imagebatch in list_of_imagebatches for im in imagebatch]

            for image in flattened:
                preds[image] = model.predict(image)[2][1].item()

            dd[path] = preds

        for imagebatch in list_of_imagebatches:
            index = imagebatch
            df_dict = {}
            for path in self.path_list:
                df_dict[path] = [dd[path][image] for image in imagebatch]

            df = pd.DataFrame.from_dict(df_dict)
            df.index = index

            imagebatch_dfs.append(df)

        return imagebatch_dfs


if __name__ == "__main__":
    path_list = ["learners/5_o_Clock_Shadow.pkl", "learners/Arched_Eyebrows.pkl"]
    mm = MultiModel(path_list)
    list_of_imagebatches = [["data/train/000334.jpg", "data/train/000335.jpg"],
                            ["data/validation/000001.jpg", "data/validation/000002.jpg"]]

    predictions = mm.predict(list_of_imagebatches)

