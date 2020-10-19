init:
	pip install --upgrade pip
	pip install --upgrade sklearn
	pip install japanize-matplotlib
	pip install pydotplus
	pip install category_encoders
	mkdir -p plugins
	mkdir -p models
	rm -rf ./plugins/*
	git clone https://github.com/Shochan024/gausian.git ./plugins/gausian
	python main.py

exec:
	rm -rf ./datas/shaped/*
	rm -rf ./datas/statistics/*
	rm -rf ./graphs/originals/*
	rm -rf ./graphs/shaped/*
	rm -rf ./graphs/statistics/*
	rm -rf graphs/models/*
	python main.py csv aggregate describe learn
