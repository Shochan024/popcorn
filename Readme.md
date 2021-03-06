# Popcorn

Popcornは、データの前処理や学習などを簡単な設定ファイルで実行できるツール

## Getting Started
好きなフォルダにcloneする
```bash
git clone https://github.com/Shochan024/popcorn.git /your_folder
```
Pythonファイルから呼び出し、初期化する
```python
import popcorn as pcn
pcn.set( work_dir="/path/to/work_dir/" )
pcn.preprocess() # 前処理
pcn.exec() # 実行 学習や推定
```
初期化により、設定ファイルとデータ格納用フォルダが生成されるので、必要情報を記述する

## Setting Files

datasには3つのフォルダが生成され、settingsには3つの設定ファイルが出力される

- datas/originals : オリジナルファイルを格納するフォルダ
- datas/shaped : 整形後のcsvが出力されるフォルダ
- datas/statistics : 統計量など集計結果が出力されるフォルダ

```bash
|-- datas
|     |-- originals
|     |-- shaped
|     |-- statistics
|-- settings
      |-- aggregate.json
      |-- csv_controll.json
      |-- graphs.json
      |-- learn.json
```

### settings/aggregate.json

csvを集計する設定ファイル
csv出力とグラフ出力の双方が伴う機能を集約

#### 1. pivot_n : pivot tableを出力する

  * agg : 集計するカラムを選択
  * mode : 集計期間を選択


```json
{
  "sample01.csv":{
    "pivot_0":{
      "agg": "[\"列1\",\"列2\",\"列3\"]",
      "mode": "year"
    },
    "pivot_1":{
      "agg": "[\"列1\",\"列2\",\"列3\"]",
      "mode": "month"
    },
    "pivot_2":{
      "agg": "[\"列1\",\"列2\",\"列3\"]",
      "mode": "day"
    }
  },
  "sample02.csv":{
    "pivot_0":{
      "agg": "[\"列1\",\"列2\",\"列3\"]",
      "mode": "month"
    }
  }
}
```

### settings/csv_controll.json

CSVを結合したり抽出する設定ファイル
CSV出力を伴う機能が集約されている

#### 1. merge : csvをmergeする

  * with : マージするcsvファイル名
  * mode : left/inner
  * on : joint_key
  * columns : 抽出する列。"all"で全列

#### 2. where : csvを抽出する

  * query : 抽出query


#### 3. describe : 基本統計量を算出

  * columns : 算出するcolumns all or array

#### 4. categorical_n : カラムをカテゴリ変数に変換する

  * columns : カテゴリ変数に変換するカラム

```json
{
  "sample01.csv":{
    "merge":{
      "with":"1_master.csv",
      "mode":"left",
      "on":"顧客ID",
      "columns": "[\"列1\",\"列2\",\"列3\"]",
    },
    "where":{
      "query":"申込日 > '2019-1-1'"
    },
    "describe":{
      "columns":"[\"列1\",\"列2\",\"列3\"]",
    },
    "categorical_0":{
      "columns":"[\"列1\",\"列2\",\"列3\"]",
    }
  }
}

```

### settings/graphs.json

グラフ出力に関する設定ファイル

#### 1. lineplot_n : 折れ線グラフを出力する

  * x : x軸
  * y : y軸

#### 2. notnull_n : 欠損値ではない割合を出力

  * x : 割合の歩幅
  * y : 出力ファイルの名前

#### 3. pairplot_n : DataFrameのpairplot

  * x : 出力するカラム
  * y : 出力ファイル名
  * hue : プロットを分けるカラム

#### 4. heatmatp_n : DataFrameのheatmap

  * x : 出力するカラム
  * y : 出力ファイル名


```json
{
  "sample01.csv":{
    "lineplot_0":{
      "x": "[\"列1\"]",
      "y": "[\"列1\"]",
    },
    "pairplot_0":{
      "x":"[\"列1\",\"列2\",\"列3\"]",
      "y": "[\"列1\",\"列2\",\"列3\"]",
      "hue" : "性別"
    },
    "heatmap_0":{
      "x":"[\"列1\",\"列2\",\"列3\"]",
      "y": "[\"列1\"]",
    }
  },
  "sample02.csv":{
    "lineplot_0":{
      "x": "[\"列1\"]",
      "y": "[\"列1\"]",
    },
    "notnull_0": {
      "x": "[0,0.01,0.05,0.1,0.3,0.6,1.0]",
      "y": "[\"欠損値ではない値\"]",
      "border": "0.6"
    }
  }
}

```

### settings/learn.json

学習モデルを実行する設定ファイル
学習モデルのdumpやグラフ出力を伴う機能が集約されている

#### 1. decisiontree_n : Decision Tree(決定木)を実行する

  * x : 説明変数
  * y : 目的変数
  * query : DataFrameを抽出するquery queryを文字列で
  * max_depth : 決定木の深さの上限 None , 1 , 2 ,,, n
  * save : グラフを保存するか

```json
{
  "sample.csv" : {
    "decisiontree_0" : {
      "x" : "[\"変数1\",\"変数2\"]",
      "y" : "[\"列1\"]",
      "query":"'2019-1-1' <= 列 <= '2019-3-31'",
      "max_depth" : "None",
      "save":"True"
    }
  }
}
```
