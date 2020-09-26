init:
	pip install --upgrade pip
	pip install --upgrade sklearn
	pip install japanize-matplotlib
	pip install pydotplus
	pip install category_encoders
	mkdir -p plugins
	rm -rf ./plugins/*
	git clone https://github.com/Shochan024/gausian.git ./plugins/gausian
	python main.py
