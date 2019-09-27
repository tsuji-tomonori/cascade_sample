# cascade_sample

## quick start

```bash
git clone https://github.com/tsuji-tomonori/cascade_sample.git
cd cascade_sample/xml
wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
cd ../
pip install -r requirements.txt
# 検出対象の画像を input フォルダーに入れる
python run.py
```

## 使用方法

pass

## xml url

* [キャラクターの顔](https://github.com/nagadomi/lbpcascade_animeface)
* [手](http://nmarkou.blogspot.com/2012/02/haar-xml-file.html)
* [opencv](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## env

```
pip install -r requirements.txt
```

動作環境

|               | version  |
| ------------- | -------- |
| Python        | 3.7.1    |
| numpy         | 1.17.2   |
| opencv-python | 4.1.1.26 |
