import sys
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

debug = False
optimization = True

# Define platform-specific compile and link arguments
if optimization:
    compile_args = {
        'msvc': ['/O2', '/fp:fast'],
        'unix': ['-Ofast', '-march=native', '-fmax-errors=1', '-fopenmp'],
        'macos': ['-Ofast', '-march=native', '-mmacosx-version-min=10.14'],
    }
else:
    compile_args = {
        'msvc': [],
        'unix': [],
        'macos': [],
    }
if debug:
    compile_args['msvc'] += ['/Zi', '/Od', '/D_DEBUG']
    compile_args['unix'] += ['-g', '-D_DEBUG']
    compile_args['macos'] += ['-g', '-D_DEBUG']


link_args = {
    'msvc': [],
    'unix': ['-fopenmp'],
    'macos': [],
}

# Determine the platform and select the appropriate arguments
if sys.platform == "win32":
    compile_args['current'] = compile_args['msvc']
    link_args['current'] = link_args['msvc']
elif sys.platform == "darwin":
    compile_args['current'] = compile_args['macos']
    link_args['current'] = link_args['macos']
else:
    compile_args['current'] = compile_args['unix']
    link_args['current'] = link_args['unix']

ext_modules = [
    Pybind11Extension(
        "l2f.interface",
        ["l2f/interface.cpp"],
        include_dirs=["external/rl-tools/include"],
        extra_compile_args=compile_args['current'],
        extra_link_args=link_args['current'],
    ),
]

setup(
    name="l2f",
    version="0.0.1",
    description="Python bindings for the L2F (Learning to Fly) Simulator",
    author="Jonas Eschmann",
    author_email="jonas.eschmann@gmail.com",
    packages=find_packages(include=['l2f']),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
