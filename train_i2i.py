from multiprocessing import Pool
import pandas as pd
import numpy as np
import glob
import tqdm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, coo_matrix
import implicit
import pickle
from copy import deepcopy
from metrics import mapk
import datetime


def get_top_overall(res):
    counter = Counter(res["product_id"])
    top50 = counter.most_common(50)
    return [item for (item, _) in top50]


def get_top_50_map(user):
    user_transactions = res[res["user_id"] == user]
    counter = Counter(user_transactions["product_id"])
    # top_50 = [item for item, _ in counter.most_common(50)]
    progress_bar.update(1)
    return counter


def get_top_50():
    p = Pool(20)
    result = p.map(get_top_50_map, all_users)
    result = dict(zip(all_users, result))
    p.close()
    p.join()
    progress_bar.close()
    return result


def get_coo_row(user):
    user_transactions = res[res["user_id"] == user]
    counter = Counter(user_transactions["product_id"])
    data = []
    j = []
    for item_id, data_item in counter.items():
        data.append(data_item)
        j.append(item_id)
    i = np.zeros_like(j)
    return coo_matrix((np.array(data).astype(np.float32), (i, j)))


def get_gt_subm_map(item):
    user_id, user_group = item
    return (user_id,  " ".join(str(id) for id in user_group['product_id']))


def get_gt_subm(res):
    pool = Pool(20)
    result = pool.map(get_gt_subm_map, tqdm.tqdm(res.groupby("user_id")))
    pool.close()
    pool.join()
    return pd.DataFrame(result, columns=['Id', "Predicted"])


def validate(submission, gt_submission):
    submission['Predicted'] = submission['Predicted'].apply(lambda x: list(map(int, x.split(" "))))
    gt_submission['Predicted'] = gt_submission['Predicted'].apply(lambda x: list(map(int, x.split(" "))))
    return mapk(gt_submission['Predicted'], submission['Predicted'], k=50)


local_validate = False

real_top_overall = [
    709,
    39590,
    166,
    55133,
    3497419,
    1300,
    67694,
    55134,
    5469728,
    3817484,
    165,
    3817542,
    158,
    63072,
    5479511,
    176,
    5217,
    69669,
    39591,
    14564,
    100789,
    9959,
    3817489,
    49911,
    3497570,
    94333,
    72875,
    7401,
    72006,
    21904,
    73725,
    687,
    14630,
    21767,
    7397,
    100,
    72003,
    68465,
    72011,
    52657,
    304,
    54728,
    39770,
    66893,
    224,
    981,
    225,
    10049,
    5642,
    100849]


def predict(row):
    _, row = row
    user_id = row["Id"]
    # if user_id in all_users:
    recommended = model.recommend(
        user_id,
        # get_coo_row(user_id).tocsr(),
        ui_matrix_train,
        N=50,
        recalculate_user=True,
        filter_already_liked_items=False,
    )
    result = recommended
    if len(result) < 50:
        result += [(id, -1) for id in real_top_overall[:50 - len(result)]]
    return result
    # else:
    #    return " ".join([str(id) for id in top_overall])


if __name__ == "__main__":
    res = pd.DataFrame()

    # Collect training data
    for fname in tqdm.tqdm(glob.glob("./sbermarket_tab_2_*/*csv")[-1:]):
        table = pd.read_csv(fname)
        res = pd.concat([res, table])

    top_overall = get_top_overall(res)

    all_users = np.unique(res["user_id"])
    all_items = np.unique(res["product_id"])

    progress_bar = tqdm.tqdm(total=len(all_users))
    # collect counters of users/item/counter
    if False:
        result = get_top_50()

        to_matrix = []
        data = []
        i = []
        j = []
        for user_id in result:
            for item_id in result[user_id]:
                i.append(user_id)
                j.append(item_id)
                data.append(result[user_id][item_id])
        # convert to a sparse matrix
        ui_matrix_train = coo_matrix((np.array(data).astype(np.float32), (i, j)))

        # save
        with open("./ui_matrix10_train.pk", "wb") as f:
            pickle.dump(ui_matrix_train, f)
    else:
        transaction_history = defaultdict(list)
        if False:
            # df = pd.read_csv("yt___tmp_damusatkina_2f363c2b_489aeb33_ed53cda5_9ffa7ca", sep="\t")

            with open("yt___tmp_damusatkina_2d8ebe55_250c0c93_5b04da79_7f99902a", "r") as f:
                i = []
                j = []
                data = []
                # df = pd.read_csv("./")
                for line in tqdm.tqdm(f, total=46194236):
                    qnt, product_id, user_id, num_orders = line.strip().split("\t")
                    qnt = float(qnt.split("=")[-1])
                    product_id = int(product_id.split("=")[-1])
                    user_id = int(user_id.split("=")[-1])
                    num_orders = float(num_orders.split("=")[-1])
                    i.append(user_id)
                    j.append(product_id)
                    data.append(num_orders)
            # ui_matrix_train = coo_matrix((np.array(df["sum_orders_decay"]).astype(np.float32), (df['user_id'].values, df['product_id'].values)))
            ui_matrix_train = coo_matrix((np.array(data).astype(np.float32), (i, j)))
            # save
            with open("./ui_matrix10_train.pk", "wb") as f:
                pickle.dump(ui_matrix_train, f)
        else:
            with open("./ui_matrix10_train.pk", "rb") as f:
                ui_matrix_train = pickle.load(f)

    if local_validate:
        # Collect validation data
        for fname in tqdm.tqdm(glob.glob("./sbermarket_tab_2_*/*csv")[-1:]):
            table = pd.read_csv(fname)
            res = pd.concat([res, table])
        # orders_info = pd.read_csv("./kaggle_tab_1345/tab_1_orders.csv")
        # res = res.merge(orders_info, how='inner')
        # res['order_created_time'] = res['order_created_time'].apply(datetime.from)

        gt_subm = get_gt_subm(res)

    # Create model and train
    for k in [100]:
        model = implicit.nearest_neighbours.TFIDFRecommender(K=k)
        if False:
            with open("./model.pk", "rb") as f:
                model = pickle.load(f)
        else:
            model.fit(ui_matrix_train.T)
        # save model
            with open("./model.pk", "wb") as f:
                pickle.dump(model, f)

        # Validate or create submission
        if local_validate:
            submission = deepcopy(gt_subm)
        else:
            submission = pd.read_csv("sample_submission.csv")
        result = []
        ui_matrix_train = ui_matrix_train.tocsr()
        pool = Pool(1)
        result = list(map(predict, tqdm.tqdm(submission.iterrows(), total=len(submission))))
        pool.close()
        pool.join()

        data = []
        for user_id, res_item in zip(submission['Id'], result):
            for item_id, conf in res_item:
                data.append((user_id, item_id, conf))

        pd.DataFrame(data, columns=['user_id', 'product_id', 'score']).to_csv("scores_tfidf_100.tsv", sep="\t")
        exit(0)
        submission["Predicted"] = result
        if local_validate:
            print(validate(submission, gt_subm))
        else:
            submission.to_csv("./submission_50.csv", index=None)
