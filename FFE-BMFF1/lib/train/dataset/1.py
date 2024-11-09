import sys
import os
import numpy as np
import glob
import argparse
from tqdm import tqdm

def main():
    prj_path = os.path.dirname(__file__)
    if prj_path not in sys.path:
        sys.path.append(prj_path)

    parser = argparse.ArgumentParser(description='Convert pytracking bbox to rgbt234 toolkit bbox format. The results are saved in "BBresults_TIP" folder')
    parser.add_argument('tracker_name', type=str, help='Name of tracker.')
    parser.add_argument('input_path', type=str, help='Path of pytracking bbox text file.')
    args = parser.parse_args()
    input_path = args.input_path
    tracker_name = args.tracker_name

    bbox_path_list = [fn for fn in glob.glob(os.path.join(input_path, '*.txt')) if not os.path.basename(fn).endswith('time.txt')]
    for bbox_path in tqdm(bbox_path_list):
        bboxs = np.loadtxt(bbox_path).tolist()
        new_bboxs = [b + b for b in bboxs]
        for r in new_bboxs:
            r2 = r.copy()
            r[2] = r2[0] + r2[2]
            r[3] = r2[1]
            r[4] = r2[0] + r2[2]
            r[5] = r2[1] + r2[3]
            r[6] = r2[0]
            r[7] = r2[1] + r2[3]
        new_bboxs = np.array(new_bboxs).astype(int)
        np.savetxt(os.path.join('BBresults_TIP', tracker_name+'_'+os.path.basename(bbox_path)), new_bboxs, delimiter=' ', fmt='%.2f')

if __name__ == '__main__':
    main()
