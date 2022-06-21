from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviar',
		sources = ['coviar_loader.c'],
		include_dirs=[np.get_include(), '/data/wanzhifan/tool/ffmpeg/include'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L/data/wanzhifan/tool/ffmpeg/lib/']
)

setup ( name = 'coviar',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
