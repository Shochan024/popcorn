init:
	pip install japanize-matplotlib
	mkdir -p plugins
	rm -rf ./plugins/*
	git clone https://github.com/Shochan024/gausian.git ./plugins/gausian
	python main.py
