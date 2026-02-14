import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import glob
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pandas as pd
import multiprocessing as mp


main_folder = os.path.normpath(rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\Defoliation\scanner')
date_folders = glob.glob(os.path.join(main_folder, '*/'))

date_folders = rf'C:\\Users\\mqalborn\\Box\\DATA_CUBBIES\\Manuel\\BLS\\Defoliation\\scanner\\20260129\\'
sample_folders = glob.glob(os.path.join(date_folders, '*/'))
sample_paths = [glob.glob(os.path.join(x, '*.png')) for x in sample_folders]
sample_paths = [item for sublist in sample_paths for item in sublist]
# sample_path = sample_paths[0]

# sample_path = rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\Defoliation\scanner/BLS_20260121_SIZE_SCAN2.jpg'

def leaf_area_scanners(path):
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y = gray.shape
    print(image.shape)
    a1 = gray.reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(a1).reshape(x, y)

    centers = kmeans.cluster_centers_.flatten()  # grayscale centers
    bg_label = np.argmax(centers)               # brightest cluster = background

    leaf_mask = (labels != bg_label).astype(np.uint8) * 255
    leaf_pixels = np.count_nonzero(leaf_mask)

    dpi = 300
    mm_per_px = 25.4 / dpi
    leaf_area_mm2 = leaf_pixels * (mm_per_px ** 2)
    leaf_area_cm2 = leaf_area_mm2 / 100.0

    # print(print(rf"{leaf_area_cm2} cm2" ))
    return leaf_area_cm2

# leaf_area_cm2 = [leaf_area_scanners(x) for x in sample_paths]

if __name__ == "__main__":
    items = sample_paths  # list of image paths
    with mp.Pool(processes=4) as pool:
        leaf_area_cm2 = pool.map(leaf_area_scanners, items, chunksize=20)

    id = [os.path.basename(x).split('.')[0] for x in sample_paths]
    folder = [x.split('\\')[-2] for x in sample_paths]

    df = pd.DataFrame({'id': id, 'sample': folder, 'leaf_area_scanner_cm2': leaf_area_cm2})

    df_grouped = (df.groupby(['sample'])
        .agg(leaf_area_scanner_cm2=('leaf_area_scanner_cm2', 'sum'))
        .reset_index())

    r = 10
    area_ref = np.pi * (r ** 2)
    df_grouped.loc[:, 'leaf_area_index_scanner'] = df_grouped.leaf_area_scanner_cm2 / area_ref

    output_path = date_folders + rf'leaf_area_scanner.xlsx'
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='leaf_area', index=False)
        df_grouped.to_excel(writer, sheet_name='LAI', index=False)
