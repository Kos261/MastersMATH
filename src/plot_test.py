import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

fig = plt.figure(figsize=(14, 8))
outer_gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[2, 1], figure=fig)

# Duży 3D
ax_big_3d = fig.add_subplot(outer_gs[:, :1], projection='3d')
ax_big_3d.set_title("Duży wykres 3D")

# Mały 3D
ax_small_3d = fig.add_subplot(outer_gs[0, 1], projection='3d')
ax_small_3d.set_title("Mały wykres 3D")

# Zagnieżdżony GridSpec (2x2) wewnątrz prawego dolnego bloku
inner_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1, 1])

ax_err_1 = fig.add_subplot(inner_gs[0, 0])
ax_err_1.set_title("Błąd 1")

ax_err_2 = fig.add_subplot(inner_gs[0, 1])
ax_err_2.set_title("Błąd 2")

ax_err_3 = fig.add_subplot(inner_gs[1, 0])
ax_err_3.set_title("Błąd 3")

ax_err_4 = fig.add_subplot(inner_gs[1, 1])
ax_err_4.set_title("Błąd 4")

plt.tight_layout()
plt.show()
