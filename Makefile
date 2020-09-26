init:
	pip install --upgrade pip
	pip install --upgrade sklearn
	pip install japanize-matplotlib
	pip install pydotplus
	mkdir -p plugins
	rm -rf ./plugins/*
	git clone https://github.com/Shochan024/gausian.git ./plugins/gausian
	python main.py
