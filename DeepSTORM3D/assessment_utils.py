# Import modules and libraries
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np


# calculate jaccard and RMSE given two arrays of xyz's and the radius for matching
# matching is done based on the hungarian algorithm, where all coords. are given in microns
def calc_jaccard_rmse(xyz_gt, xyz_rec, radius):
    
    # if the net didn't detect anything return None's
    if xyz_rec is None:
        print("Empty Prediction!")
        return 0.0, None, None, None
    
    else:
        
        # calculate the distance matrix for each GT to each prediction
        C = pairwise_distances(xyz_rec, xyz_gt, 'euclidean')
        
        # number of recovered points and GT sources
        num_rec = xyz_rec.shape[0]
        num_gt = xyz_gt.shape[0]
        
        # find the matching using the Hungarian algorithm
        rec_ind, gt_ind = linear_sum_assignment(C)
        
        # number of matched points
        num_matches = len(rec_ind)
        
        # run over matched points and filter points radius away from GT
        indicatorTP = [False]*num_matches
        for i in range(num_matches):
            
            # if the point is closer than radius then TP else it's FP
            if C[rec_ind[i], gt_ind[i]] < radius:
                indicatorTP[i] = True
        
        # resulting TP count
        TP = sum(indicatorTP)

        # resulting jaccard index
        jaccard_index = TP / (num_rec + num_gt - TP)
        
        # if there's TP
        if TP:
            
            # pairs of TP
            rec_ind_TP = (rec_ind[indicatorTP]).tolist()
            gt_ind_TP = (gt_ind[indicatorTP]).tolist()
            xyz_rec_TP = xyz_rec[rec_ind_TP, :]
            xyz_gt_TP = xyz_gt[gt_ind_TP, :]
            
            # calculate mean RMSE in xy, z, and xyz
            RMSE_xy = np.sqrt(np.mean(np.sum((xyz_rec_TP[:,:2] - xyz_gt_TP[:,:2])**2, 1)))
            RMSE_z = np.sqrt(np.mean(np.sum((xyz_rec_TP[:,2:] - xyz_gt_TP[:,2:])**2, 1)))
            RMSE_xyz = np.sqrt(np.mean(np.sum((xyz_rec_TP - xyz_gt_TP)**2, 1)))
            
            return jaccard_index, RMSE_xy, RMSE_z, RMSE_xyz
        else:
            return jaccard_index, None, None, None

