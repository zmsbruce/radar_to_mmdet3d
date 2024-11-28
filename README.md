# radar_to_mmdet3d

从 Robomaster Radar 的点云序列文件和视频生成 MMDetection 3D 数据集

## 编译和运行

```sh
# 拉取仓库文件和代码
git clone --depth 1 https://github.com/zmsbruce/radar_to_mmdet3d.git
git lfs pull

# 安装依赖
sudo apt install clang libavcodec-dev libavformat-dev libavutil-dev pkg-config

# 编译
cargo build --release

# 运行测试
cargo test

# 运行程序
cargo run --release
```

## 配置文件

- [radar.toml](config/radar.toml) 配置了三个相机实例的内参和激光雷达与相机之间的转换矩阵，以及检测和定位相关的参数；
- [source.toml](config/source.toml) 配置了点云数据文件路径、输出目录路径、和多个视频路径；

## TODO

- 添加相机内外参的写入；
- 为 YOLO 检测支持批量处理，以提升速度；
- 对定位做改进以提升速度，如深度图更新只添加检测框内的点；
- 完善日志记录；
- 添加注释；