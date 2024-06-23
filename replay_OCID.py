import numpy as np
method_id = 'GT_bb_sam'
dataset = 'OCID'
result = np.load('{}_{}_results.npy'.format(dataset, method_id))

print(method_id)
print('Overlap Prec:{:.4f}, Rec:{:.4f}, F_score:{:.4f}, Boundary Prec:{:.4f}, Rec:{:.4f}, F_score:{:.4f}, %75:{:.4f}'. \
    format(np.mean(result[:, 1]), np.mean(result[:, 2]), np.mean(result[:, 0]),
            np.mean(result[:, 4]), np.mean(result[:, 5]), np.mean(result[:, 3]), np.mean(result[:, 6])))