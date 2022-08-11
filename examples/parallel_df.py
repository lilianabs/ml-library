# from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

# If you have a big dataframe and a function that cant be easily vectorized
# then you ccan run it in parallel.
# https://twitter.com/marktenenholtz/status/1557336004721160192/photo/1


def my_func(row):
    return row["col"] ** 2


if __name__ == "__main__":

    df = pd.DataFrame({"col": list(range(30000))})

    num_jobs = cpu_count()

    print(f"Number of cores {num_jobs}")

    with Pool(num_jobs) as p:

        # split your df into chuncks
        split_df = np.array_split(df, num_jobs)

        # map your function to each split it parallel
        # and concat them back together
        res = p.map(my_func, split_df)
        res = pd.concat(res)

    print(res)
