# readme

介绍+插图

## 数据预处理

### UCF101数据处理：

1、从[官网](https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar)下载UCF101数据集，解压到data文件夹下；

2、安装FFmpeg工具（推荐版本3.1.3，新版本可能导致无法正常安装dataloader），下载地址在[这里](http://www.ffmpeg.org/releases/)：

(1)、从网址下载需要的安装包，解压到需要的路径下，不建议在安装包所在的路径中解压；

(2)、由于是从官方库中直接下载的老版本安装包，省略其他教程中可能存在的check版本操作：

```
git checkout 74c6a6d3735f79671b177a0e0c6f2db696c2a6d2.
```

(3)、在解压好的FFmpeg路径下执行如下命令：

```
make clean
./configure --prefix=${FFMPEG_INSTALL_PATH} --enable-pic --disable-yasm --enable-shared
make
make install
```

(4)、此时应该就可以运行了，但如果权限不足可能还需要修改bashrc中的几个条目，将FFmpeg目录`${FFMPEG_INSTALL_PATH}/lib/`添加到相应的路径下`$LD_LIBRARY_PATH`，例如：

```
export PATH=$PATH:"/tool/ffmpeg/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/tool/ffmpeg/lib"
```

3、使用FFmpeg处理avi格式的视频到mp4格式；

```
sh reencode.sh ucf101/UCF-101/ ucf101/UCF-101_rawvideo/
```

/data文件夹中的结构如下：

```
ucf101
  ├── UCF-101/
  ├── UCF-101_rawvideo/
  └── ucfTrainTestlist/
```

ucfTrainTestlist中存放的是做动作识别的trainlist和testlist，同样可在[这个页面](https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip)下载；

4、为了方便数据处理，在这里使用coviar的dataloader：

首先更改`ops/coviar_loader/setup.py`文件中的内容，`${FFMPEG_INSTALL_PATH}`指向自己的ffmpeg路径即可（不建议使用conda自带的ffmpeg，可以通过修改~/.bashrc中的路径来实现）：

```
coviar_utils_module = Extension('coviar',
		sources = ['coviar_loader.c'],
		include_dirs=[np.get_include(), '${FFMPEG_INSTALL_PATH}/include'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L/${FFMPEG_INSTALL_PATH}/lib/']
)
```

在`steup.py`完成之后运行脚本进行安装即可。

```
cd ops/coviar_loader
sh install.sh
```

5、环境配置：

```
python3.8，
CUDA version == 11.3
cuDNN version is 8200
torch version ==  1.10.0
opencv version == 4.5.5
numpy
decord==0.4.0
Pillow
```

6、运行代码

预训练参数见`cfg/ucf_config.py`，然后通过脚本运行，支持多卡，脚本中可以设置`CUDA_VISIBLE_DEVICES`。

```
sh pretrain.sh
```

微调参数见`cfg/finetune_ucf_config.py`，然后通过脚本运行，暂不支持多卡训练。

```
sh finetuen.sh
```

### 暴力分拣数据集

修改中
