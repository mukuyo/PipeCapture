import numpy as np

# 正解の回転行列と推定した回転行列
R_true = np.array([
    [-0.97756, 0.20778, -0.034641],
    [0.082325, 0.22548, -0.97076],
    [-0.19389, -0.95183, -0.23753]
])

R_estimated = np.array([
    [0.21937, -0.97207, -0.083411],
    [0.03472, -0.077661, -0.99637],
    [0.97502, 0.22147, 0.016714]
])

# 1. 行列式の計算
det_true = np.linalg.det(R_true)
det_estimated = np.linalg.det(R_estimated)
print(f"正解行列の行列式: {det_true}")
print(f"推定行列の行列式: {det_estimated}")

# 2. 2つの行列の積を計算し、単位行列に近いかどうかを確認
R_difference = np.dot(R_true, R_estimated.T)
print("2つの行列の積:\n", R_difference)
identity_check = np.allclose(R_difference, np.eye(3), atol=1e-2)
print(f"単位行列に近いか: {identity_check}")

# 3. 回転軸と回転角の計算
def rotation_axis_angle(R):
    # 回転角度の計算
    angle = np.arccos((np.trace(R) - 1) / 2)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis, np.degrees(angle)

axis_true, angle_true = rotation_axis_angle(R_true)
axis_estimated, angle_estimated = rotation_axis_angle(R_estimated)

print(f"正解行列の回転軸: {axis_true}, 回転角度: {angle_true}度")
print(f"推定行列の回転軸: {axis_estimated}, 回転角度: {angle_estimated}度")
