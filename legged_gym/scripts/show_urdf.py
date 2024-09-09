import pybullet as p
import pybullet_data
import time

# 连接到 PyBullet 仿真环境
p.connect(p.GUI)

# 设置 PyBullet 数据路径（包含一些示例 URDF 文件）
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 添加地面
plane_id = p.loadURDF("plane.urdf")

# 加载 URDF 文件，这里使用的是 PyBullet 提供的示例文件
robot_id = p.loadURDF("/home/xingxing/repos/legged_gym/resources/robots/tinker/urdf/tinker_urdf.urdf")

# 或者加载你自己的 URDF 文件，例如：
# robot_id = p.loadURDF("/path/to/your/robot.urdf")

# 设置仿真环境的重力
p.setGravity(0, 0, -9.81)

# 运行仿真主循环
while True:
    # 步进仿真
    p.stepSimulation()
    time.sleep(1./240.)  # 设置仿真步长
