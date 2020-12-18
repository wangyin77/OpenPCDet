import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():                            # 获取当前版本号
    if not os.path.exists('.git'):                      # 如果没有.git,版本号就是000000
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
                                                        # 
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):        # 创建了一个pcdet/version.py文件
    with open(target_file, 'w') as f:                   # 以只写形式打开
        print('__version__ = "%s"' % version, file=f)   # 写入__version__ = （版本号）


if __name__ == '__main__':
    version = '0.3.0+%s' % get_git_commit_number()      # 通过.git查看当前版本
    write_version_to_file(version, 'pcdet/version.py')  # 创建文件，写入版本号

    setup(                                              # 通过setuptools打包工具，打包成pcdet库
        name='pcdet',                                   # 库的名字
        version=version,                                # 库的版本
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',     # 库的描述
        install_requires=[                              # 自动安装一些依赖
            'numpy',
            'torch>=1.1',
            'spconv',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        author='Shaoshuai Shi',                         # 库的作者
        author_email='shaoshuaics@gmail.com',           # 作者邮箱    
        license='Apache License 2.0',                   # 开源协议
        packages=find_packages(exclude=['tools', 'data', 'output']),
                                                        # 通过find——packas选择要打包的python包（含有__init__.py的文件，内容为空也可）
                                                        # 其实就是指定打包OpenPcdet文件夹下的pcdet文件夹中的内容
        '''
        添加自定义命令build_ext，来自torch.utils.cpp_extension
        //网上的解释：当使用BuildExtension时，它将提供一个用于extra_compile_args（不是普通列表）的词典，通过语言（cxx或cuda）映射到参数列表提供给编译器。
            这样可以在混合编译期间为C++和CUDA编译器提供不同的参数。
        我觉得就是在下面增加扩展ext的时候，调用CUDA编译cpp文件，而不是用g++编译。
        '''
        cmdclass={'build_ext': BuildExtension},         
        ext_modules=[                                   # 指定扩展模块，调用make_cuda_ext函数
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
        ],
    )
