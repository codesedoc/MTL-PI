import utils.hierarchical as hierarchical


h = hierarchical.Hierarchic()
# h._pretrain_model(h.coherence_controller, h.coherence_model_path)
# h._pretrain_model(h.resemblance_controller, h.resemblance_model_path)
h._test_models()


# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# colors = ['white']
# cmap = mpl.colors.ListedColormap(colors)
# m = np.array([[1,2,3],[4,5,6],[7,8,9]])
# plt.matshow(m, cmap=plt.get_cmap('binary'))
# plt.show()
