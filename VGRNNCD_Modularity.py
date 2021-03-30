import matplotlib.pyplot as plt

x_data = [0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.036]
y_data = [64.5469, 65.0405, 64.8070, 65.2967, 66.0334, 65.0194, 65.0341, 65.7271, 65.3299]

#voles
plt.plot(x_data, y_data, color='red', linewidth = 2.0, linestyle='--')
plt.title("Q varies from patameters in voles", fontproperties="songti")  # 设置标题及字体
plt.show()