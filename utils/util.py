from __future__ import print_function

import torch
import numpy as np
import pymssql
import sqlite3
import os
from typing import Tuple

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size


def local_db_backup(insert_sql):
    conn = sqlite3.connect('./Quantization.db')
    cursor_obj = conn.cursor()

    cursor_obj.execute('''CREATE TABLE IF NOT EXISTS quantization (model_name, pid, dataset, parameter, type, bits, distillation, metric_1, metric_2, accuracy, metric_3, metric_4, parameter_desc, metric_1_desc, metric_2_desc, metric_3_desc, metric_4_desc)''')
    conn.commit()

    cursor_obj.execute(insert_sql)
    conn.commit()
    conn.close()
    
    
def insert_SQL(model_name, pid, dataset, parameter_desc, parameter, type_q, bits, distillation, accuracy, metric_1_desc, metric_1, metric_2_desc, metric_2, metric_3_desc, metric_3, metric_4_desc, metric_4):
    insert_sql = """INSERT into quantization (model_name, pid, dataset, parameter, type, bits, distillation, metric_1, metric_2, accuracy, metric_3, metric_4, parameter_desc, metric_1_desc, metric_2_desc, metric_3_desc, metric_4_desc) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')""".format(model_name, pid, dataset, parameter, type_q, bits, distillation, metric_1, metric_2, accuracy, metric_3, metric_4, parameter_desc, metric_1_desc, metric_2_desc, metric_3_desc, metric_4_desc)
    
    try:
        lines = []

        with open('MSSQL.txt') as f:
            lines = f.read().splitlines()

        connSQL = pymssql.connect(server=lines[0], user=lines[1], password=lines[2], database=lines[3], port = 8001)
        cursorSQL = connSQL.cursor()
        cursorSQL.execute(insert_sql)
        connSQL.commit()
        connSQL.close()
    except Exception as e:
        # Local backup
        local_db_backup(insert_sql)
        print("Results not inserted in MSSQL")

def get_free_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r+').readlines()]
        id = np.argmax(memory_available)
        device = device + ':' + str(id)
        os.system('rm tmp')
    return device


def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(y_true.shape) > 1:
        return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

    else:
        return y_true, (y_preds > 0.5).astype(int)

if __name__ == '__main__':

    pass
