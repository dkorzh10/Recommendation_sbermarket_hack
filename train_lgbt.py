import pandas as pd
from multiprocessing import Pool
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')


def predict_mapper(user_id):
    user_id = int(user_id)
    subrow = dfall[dfall['user_id'] == user_id]
    to_predict = subrow[['score_x', 'score_y', 'score']]
    subrow.loc[:, 'prediction'] = (model.predict(to_predict.values) + model3.predict(to_predict.values)) / 2
    subrow.loc[:, :] = subrow.sort_values("prediction", ascending=False)
    return (user_id, " ".join(str(x) for x in subrow['product_id']))


if __name__ == "__main__":
    dfall = pd.read_csv("./scores_for_all_pairs.tsv", sep="\t")
    uioccur = pd.read_csv("yt___tmp_damusatkina_2f363c2b_489aeb33_ed53cda5_9ffa7ca", sep="\t")

    print("Creating X")
    X = dfall.merge(uioccur, on=['user_id', 'product_id'], how="inner")
    y = X['sum_orders_decay']
    X = X.drop('sum_orders_decay', axis=1)
    print("Train test split")
    X_train, X_test, y_train, y_test = train_test_split(X[['score_x', 'score_y', 'score']].values, y.values)

    if False:
        model = LGBMRegressor(n_estimators=200)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        with open("./model_lgb.pk", "wb") as f:
            pickle.dump(model, f)
    else:
        with open("./model_lgb.pk", "rb") as f:
            model = pickle.loads(f.read())

    if False:
        if False:
            model2 = CatBoostRegressor(iterations=1000, learning_rate=0.1)
            model2.fit(X_train, y_train)
            with open("./model_cb.pk", "wb") as f:
                pickle.dump(model2, f)
        else:
            with open("./model_cb.pk", "rb") as f:
                model2 = pickle.load(f)

    if False:
        model3 = XGBRegressor(n_estimators=1000, verbosity=2)
        model3.fit(X_train, y_train)
    else:
        with open("./model_xgb.pk", "rb") as f:
            model3 = pickle.loads(f.read())

    pool = Pool(20)
    all_ids = []
    with open("./sample_submission.csv", "r") as f:
        for line in f:
            id, _ = line.strip().split(",")
            all_ids.append(id)
    all_ids = all_ids[1:]
    result = list(map(predict_mapper, tqdm.tqdm(all_ids)))
    pd.DataFrame(result, columns=["Id", "Predicted"]).to_csv("./submission_boosting.csv", index=None)
