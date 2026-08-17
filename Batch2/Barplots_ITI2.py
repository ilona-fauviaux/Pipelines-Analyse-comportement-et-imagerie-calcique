# import matplotlib.pyplot as plt

# from Program_A import (
#     modeli_go,
#     modeli_go_touch,
#     modeli_nogo,
#     modeli_nogo_touch,
# )

# models = {
#     "Go": modeli_go,
#     "Go-Touch": modeli_go_touch,
#     "Nogo": modeli_nogo,
#     "Nogo-Touch": modeli_nogo_touch
# }

# plt.figure(figsize=(7,4))

# y = 0

# labels = []

# for name, model in models.items():

#     if model is None:
#         continue

#     coef = model.params["ITI2"]

#     conf = model.conf_int()

#     lower = conf.loc["ITI2", 0]
#     upper = conf.loc["ITI2", 1]

#     # IC95%
#     plt.plot(
#         [lower, upper],
#         [y, y],
#         color="gray",
#         linewidth=2
#     )

#     plt.plot(
#         coef,
#         y,
#         "o",
#         color="black",
#         markersize=8
#     )

#     labels.append(name)

#     y += 1

# plt.axvline(
#     0,
#     color="black",
#     linestyle="--"
# )

# plt.yticks(range(len(labels)), labels)

# plt.xlabel("Coefficient de ITI2")
# plt.title(
#     "Effet de ITI2 sur la performance\nToutes souris et sessions confondues pour Batch2"
# )

# plt.tight_layout()
# plt.show()