SHELL := /bin/bash

build:
	python3 -m build

install:
	python3 -m pip install ./dist/remin-*.tar.gz

ebuild:
	python3 -m pip install -e .

uninstall:
	python3 -m pip uninstall .

tree:
	tree --gitignore .