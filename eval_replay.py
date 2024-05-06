import numpy as np
result_file = 'GT_bb_sam'
result = np.load('{}_results.npy'.format(result_file))
print(result.shape)
result_dict = dict()
result_dict['seen'] = result[:30]
result_dict['similar'] = result[30:60]
result_dict['novel'] = result[60:]

for split, result in result_dict.items():
    print('{} Prec:{:.4f}, Rec:{:.4f}, F_score:{:.4f}, Boundary Prec:{:.4f}, Rec:{:.4f}, F_score:{:.4f}'. \
        format(split, np.mean(result[:, :, 1]), np.mean(result[:, :, 2]), np.mean(result[:, :, 0]),
                np.mean(result[:, :, 4]), np.mean(result[:, :, 5]), np.mean(result[:, :, 3])))