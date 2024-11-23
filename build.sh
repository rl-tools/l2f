# auth in .pypirc
(cd external/rl-tools/static/ui_server/generic && ./download_dependencies.sh)
rm -rf ../l2f/dist
pip install --upgrade build twine
python3 -m build --sdist
python3 -m twine upload dist/*