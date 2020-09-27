# Popcorn

Popcornは、データの前処理や学習などを簡単な設定ファイルで実行できるツールである。

## Getting Started
好きなフォルダにcloneする
```bash
git clone https://github.com/Shochan024/popcorn.git /your_folder
```
Pythonファイルから呼び出し、初期化する
```python
import popcorn as pcn
pcn.set( data_path="/path/to/datapath/" , setting_path="/path/to/settingpath" )
pcn.preprocess()
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

#### 1. pivot_n : pivot tableを出力する

  * agg : 集計するカラムを選択
  * mode : 集計期間を選択

#### 2. categorical_n : pivot tableを出力する

  * columns : カテゴリ変数に変換するカラム

```json
{
  "sample01.csv":{
    "pivot_0":{
      "agg": "[\"契約日/契約予定日\",\"決済ステータス\",\"決済ステータスID\"]",
      "mode": "year"
    },
    "pivot_1":{
      "agg": "[\"契約日/契約予定日\",\"集客経路\",\"決済ステータスID\"]",
      "mode": "month"
    },
    "pivot_2":{
      "agg": "[\"契約日/契約予定日\",\"性別\",\"決済ステータスID\"]",
      "mode": "day"
    },
    "categorical_0":{
      "columns":"[\"性別\",\"集客経路\"]"
    }
  },
  "sample02.csv":{
    "pivot_0":{
      "agg": "[\"契約日/契約予定日\",\"決済ステータス\",\"決済ステータスID\"]",
      "mode": "month"
    }
  }
}
```

### settings/csv_controll.json

CSVを結合したり抽出する設定ファイル

#### 1. merge : csvをmergeする

  * with : マージするcsvファイル名
  * mode : left/inner
  * on : joint_key
  * columns : 抽出する列。"all"で全列

#### 2. where : csvを抽出する

  * query : 抽出query


#### 3. describe : 基本統計量を算出

  * columns : 算出するcolumns all or array

```json
{
  "sample01.csv":{
    "merge":{
      "with":"1_master.csv",
      "mode":"left",
      "on":"顧客ID",
      "columns": "[\"顧客ID\",\"申込日\",\"契約日/契約予定日\",\"希望決済日\"]"
    },
    "where":{
      "query":"申込日 > '2019-1-1'"
    },
    "describe":{
      "columns":"[\"契約時年齢\",\"契約時年収\",\"定価\"]"
    }
  }
}

```

### settings/graphs.json

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
      "x": "[\"契約日/契約予定日\"]",
      "y": "[\"性別\"]"
    },
    "pairplot_0":{
      "x":"[\"性別\",\"契約時年齢\",\"契約時年収\",\"最低販売価格\",\"融資金額\",\"支払金額\"]",
      "y": "[\"pairplot_性別\"]",
      "hue" : "性別"
    },
    "heatmap_0":{
      "x":"[\"契約時年齢\",\"契約時年収\",\"最低販売価格\",\"融資金額\",\"支払金額\"]",
      "y": "[\"heatmap\"]"
    }
  },
  "sample02.csv":{
    "lineplot_0":{
      "x": "[\"契約日/契約予定日\"]",
      "y": "[\"決済ステータス\"]"
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
      "y" : "[\"決済ステータスID\"]",
      "query":"'2019-1-1' <= 申込日<= '2019-3-31'",
      "max_depth" : "None",
      "save":"True"
    }
  }
}
```
