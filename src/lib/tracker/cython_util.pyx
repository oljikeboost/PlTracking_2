from __future__ import print_function
import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_img
DTYPE = np.float
ctypedef np.float_t DTYPE_t

def crop_images(list all_results, list inp_data, np.int offset):

    cdef float max_prob
    cdef np.ndarray img, jersey_crop
    cdef list result
    cdef int h,w, length
    # cdef np.ndarray max_res = np.zeros((5), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=1] max_res = np.zeros((5), dtype=DTYPE)
    ## How to define img, jersey_crop, result, max_res

    cdef int x1, x2, y1, y2

    output = []
    for idx, result in enumerate(all_results):

        length = len(result[0])
        if length == 0:
            continue

        img = inp_data[idx]

        max_res = max(result[0], key=lambda x: x[-1])
        max_prob = max_res[-1]
        if max_prob < 0.6:
            continue

        # max_res = max_res.astype(np.int)
        x1 = int(max_res[0])
        x2 = int(max_res[2])
        y1 = int(max_res[1])
        y2 = int(max_res[3])

        jersey_crop = img[y1 - offset: y2 + offset,
                      x1 - offset: x2 + offset, :]

        h = jersey_crop.shape[0]
        w = jersey_crop.shape[1]
        if 0 in (h,w):
            jersey_crop = img[y1:y2, x1:x2, :]

        h = jersey_crop.shape[0]
        w = jersey_crop.shape[1]
        if 0 in (h,w):
            continue

        output.append(jersey_crop)

    return jersey_crop