import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_input_vols(v, m, e):

    v.index = m
    v.columns = e
    # n_lines = len(v.columns)
    # c = np.arange(1, n_lines+1)

    # norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    # cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    # cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)
    for i in range(len(v.columns)):
        ax.plot(v.index, v.iloc[:, i], label='T=' + str(round(e[i], 2)), linewidth=0.6)
        # ax.plot(v.index, v.iloc[:, i], c=cmap.to_rgba(len(v.columns) - i + 2), label=round(e[i], 2))
    # fig.colorbar(cmap, c)
    plt.xlabel('log moneyness')
    plt.ylabel('IV')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('SPX IV for 24/07/2024')
    plt.savefig(r'C:\Users\axelc\OneDrive\Documents\PY\ABRC\output\plots\iv_spx.jpg')
    plt.show()

    # colors = [10*i for i in range(len(m))]
#     plt.figure(figsize=(5, 5))
#     for i in range(len(v.columns)):
#         plt.scatter(v.index, v.iloc[:, i], linewidths=0.3, marker='.')#, c=colors, cmap='inferno') #viridis
#         plt.plot(v.index, v.iloc[:, i], linewidth=1)
#     plt.colorbar(mpl.cm.S)
#     plt.xlabel('K')
#     plt.grid(True)
#     plt.title('SPX IV for 24/07/2024')
# #    plt.rc('grid', linestyle='-', color='black')
#     plt.show()