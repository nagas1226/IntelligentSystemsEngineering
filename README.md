# Vehicle Price Prediction Project

中古車価格予測のための機械学習プロジェクトです。

## データセットの準備

データセットは[LMS](https://lms.s.isct.ac.jp) からダウンロードし，`dataset/` フォルダに配置してください．

## プロジェクト構成

```
topicA/
├── dataset/                          # データセット
│   ├── projectA_vehicle_train.csv
│   ├── projectA_vehicle_val.csv
│   └── projectA_vehicle_test.csv
├── src/                              # ソースコード
│   ├── config/                       # 設定管理
│   │   ├── base_config.py           # 基本設定クラス
│   │   └── default_config.py        # デフォルト設定
│   ├── features/                     # 特徴量エンジニアリング
│   │   ├── base_encoder.py          # 基底エンコーダークラス
│   │   ├── preprocess.py            # 前処理パイプライン
│   │   ├── target_encoding.py       # ターゲットエンコーディング
│   │   └── *.py                     # 各種エンコーダー
│   └── metrics.py                   # 評価指標
├── notebooks/                        # Jupyter notebooks
│   ├── *
└── README.md                        # このファイル
```
