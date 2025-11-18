import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xs, ys, zs = [], [], []

with open("/Users/kimminseong/Desktop/UNIV/UNIV 3-2/기계학습/dataset/machine_learning1/vertical/4.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        cols = line.split(",")          # 콤마 기준으로 자르기
        if len(cols) <= 6:
            continue
        xyz_str = cols[6].strip()       # 6번 인덱스
        # 예: "155/-585/-174"
        parts = xyz_str.split("/")
        if len(parts) != 3:
            continue
        x, y, z = map(float, parts)
        xs.append(x)
        ys.append(y)
        zs.append(z)


# 3D 궤적 애니메이션: 시계열 순서대로 한 점씩 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory Marker")
ax.grid(True)

# 전체 범위를 미리 잡아서 화면이 흔들리지 않도록 설정
if xs and ys and zs:
    margin = 10  # 여유 여백
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin
    zmin, zmax = min(zs) - margin, max(zs) + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

line3d, = ax.plot([], [], [], "-o", markersize=3)

for i in range(len(xs)):
    line3d.set_data(xs[: i + 1], ys[: i + 1])
    line3d.set_3d_properties(zs[: i + 1])
    plt.draw()
    plt.pause(0.02)  # 속도 조절 (초 단위)

plt.show()

# 탑뷰(XY) 애니메이션: 시계열 순서대로 한 점씩 그리기
fig2, ax2 = plt.subplots()
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Top View (XY) - Time-ordered Animation")
ax2.axis("equal")
ax2.grid(True)

# 전체 범위를 미리 잡아서 화면이 흔들리지 않도록 설정
if xs and ys:
    margin = 10  # 여유 여백
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

line, = ax2.plot([], [], "-o", markersize=3)

# 시계열 순서대로 한 점씩 추가하면서 그리기
for i in range(len(xs)):
    line.set_data(xs[: i + 1], ys[: i + 1])
    plt.draw()
    plt.pause(0.02)  # 속도 조절 (초 단위)

# 애니메이션이 끝난 후 최종 궤적 유지
plt.show()

