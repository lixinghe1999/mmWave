from mmpose.apis import MMPoseInferencer
import os
import json
import numpy as np
inferencer = MMPoseInferencer('human')
data = os.listdir('data')
keypoints = {}
for d in data:
    for f in os.listdir('data/' + d):
        if f[0] == 'R':
            img_path = os.path.join(
                'data', d, f
            )
            # The MMPoseInferencer API employs a lazy inference approach,
            # creating a prediction generator when given input
            result_generator = inferencer(img_path, show=True)
            result = next(result_generator)
            ps = []
            for p in result['predictions'][0]:
                ps.append(p['keypoints'])
            ps = np.concatenate(ps)
            np.save('data/' + d + '/K' + f[1:] + '.npy', ps)
            break
