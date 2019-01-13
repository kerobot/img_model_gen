# img_model_gen

To Categorical And Save Model by Keras

画像ファイルをもとに学習データとテストデータのラベル付けおよびモデルの保存を行います。

## 環境

* Windows 10 x64 1809
* Python 3.6.5 x64
* Power Shell 6 x64
* Visual Studio Code x64
* Git for Windows x64

## 構築

プロジェクトを clone してディレクトリに移動します。

```powershell
> git clone https://github.com/kerobot/img_model_gen.git img_model_gen
> cd img_model_gen
```

プロジェクトのための仮想環境を作成して有効化します。

```powershell
> python -m venv venv
> .\venv\Scripts\activate.ps1
```

念のため、仮想環境の pip をアップグレードします。

```powershell
> python -m pip install --upgrade pip
```

依存するパッケージをインストールします。

```powershell
> pip install -r requirements.txt
```

## 実行

face_scratch_imageディレクトリに学習データ、test_imageディレクトリにテストデータとなる画像ファイルを配置して実行します。

> モデルファイルの出力

```powershell
> python .\img_model_gen.py
```
