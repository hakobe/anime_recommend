# 今季見るべきアニメ推薦くん

くわしくはこちら:http://hakobe932.hatenablog.com/entry/2016/04/15/142756

## 動かし方

```shell
# 1. Python環境の準備
#   pyenv + miniconda3 をおすすめします
$ pyenv install miniconda3-3.19.0
$ pyenv local miniconda3-3.19.0
$ conda install numpy
$ conda install sklearn
$ conda install scipy
$ conda install chainer
$ conda install matplotlib
#
# 2. アニメの評価
#   気にいったアニメなら y そうでなければ何も入力せずにEnter
#   Ctrl-C で途中で止められます
$ python eval_animes.py
#
# 3. しょぼかるからアニメ情報を取得
$ python collect_anime_infos.pya
#
# 4. アニメ情報から特徴を生成
$ python convert_to_features.py
#
# 5. 推薦を実行
#   少し時間がかかります
$ python anime_recommend.py
```
