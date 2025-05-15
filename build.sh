# auth in .pypirc
(cd external/rl-tools/static/ui_server/generic && ./download_dependencies.sh)
(cd external/json/nlohmann && wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp)
rm -rf ../l2f/dist
pip install --upgrade build twine
python3 -m build --sdist
python3 -m twine upload dist/*
