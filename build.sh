rm -rf ../python-interface/dist
pip install --upgrade build twine
python3 -m build --sdist
# python3 -m twine upload dist/* --password $PYPI_TOKEN