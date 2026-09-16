目录结构：
Real-ESRGAN-latest：修改自 https://github.com/xinntao/Real-ESRGAN.git 项目，增加调用接口代码及增添RoSE项目相关配置。完整项目，包含各种测试等。其中无人机项目有调用的部分也复制于/RoSE/sim/目录下，如inference_realesrgan_reorg.py
RoSE：项目主模块。包含FireSim硬件模块与FireSim_Mock仿真模块等，调用到Real-ESRGAN-latest等项目的接口。这里将主要对RoSE项目的增添整理于/sim, /deploy/hephaestus几个目录下。其中demo图形可见/deploy/hephaestus，received_img_xxx.png为原项目无人机图片接收，received_img_sr_xxx.png为接入超分算法后无人机图片处理结果。
UAV-DRL：源自 https://github.com/heleidsn/UAV_Navigation_DRL_AirSim.git 项目，对于本项目进行专门调试和训练。
