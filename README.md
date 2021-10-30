# variational-auto-encoder

# Auto Encoder

## description

>画像の圧縮と復元を通じて，入力する画像に近い画像を復元するモデル．  
>ネットワークは入力層，隠れそうは同じ次元数，隠れ層の次元数は上方を圧縮するため入力数より小さい次元に圧縮する．  
>入力層と隠れ層は画像を圧縮するエンコーダと言われる．  
>隠れそうと出力層は画像を復元するためデコーダと呼ばれる．  
>AutoEncoderの学習は入力画像xと復元画像x'をピクセル単位で比較．そのため，入力層の次元数は画像解像度（高さ，幅）* チャネル数になります．  
>入力画像と復元画像の画素値を損失関数で比較する．（再構成誤差）  

[引用先](https://www.shuwasystem.co.jp/book/9784798062297.html)  

### Reconstruction Error

<a href="https://www.codecogs.com/eqnedit.php?latex=J^{REG}&space;=&space;-\frac{1}{N}&space;\sum_{i}^{N}(x_i&space;\log(y_i)&space;&plus;&space;(1-x_i)&space;\log(1-y_i))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J^{REG}&space;=&space;-\frac{1}{N}&space;\sum_{i}^{N}(x_i&space;\log(y_i)&space;&plus;&space;(1-x_i)&space;\log(1-y_i))" title="J^{REG} = -\frac{1}{N} \sum_{i}^{N}(x_i \log(y_i) + (1-x_i) \log(1-y_i))" /></a>

# Reference
- https://github.com/ayukat1016/gan_sample
