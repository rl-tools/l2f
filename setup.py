import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define platform-specific compile and link arguments
compile_args = {
    'msvc': ['/O2', '/fp:fast'],
    'unix': ['-Ofast', '-march=native'],
    'macos': ['-Ofast', '-march=native', '-mmacosx-version-min=10.14'],
}

link_args = {
    'msvc': [],
    'unix': [],
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
        "l2f",
        ["src/l2f.cpp"],  # Adjust the source file paths as necessary
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
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
