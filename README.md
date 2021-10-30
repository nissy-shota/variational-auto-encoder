# AutoEncoder

## Description

>画像の圧縮と復元を通じて，入力する画像に近い画像を復元するモデル．  
>ネットワークは入力層，隠れそうは同じ次元数，隠れ層の次元数は上方を圧縮するため入力数より小さい次元に圧縮する．  
>入力層と隠れ層は画像を圧縮するエンコーダと言われる．  
>隠れそうと出力層は画像を復元するためデコーダと呼ばれる．  
>AutoEncoderの学習は入力画像xと復元画像x'をピクセル単位で比較．そのため，入力層の次元数は画像解像度（高さ，幅）* チャネル数になります．  
>入力画像と復元画像の画素値を損失関数で比較する．（再構成誤差）  

[引用先](https://www.shuwasystem.co.jp/book/9784798062297.html)  

### Reconstruction Error

<a href="https://www.codecogs.com/eqnedit.php?latex=J^{REG}&space;=&space;-\frac{1}{N}&space;\sum_{i}^{N}(x_i&space;\log(y_i)&space;&plus;&space;(1-x_i)&space;\log(1-y_i))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J^{REG}&space;=&space;-\frac{1}{N}&space;\sum_{i}^{N}(x_i&space;\log(y_i)&space;&plus;&space;(1-x_i)&space;\log(1-y_i))" title="J^{REG} = -\frac{1}{N} \sum_{i}^{N}(x_i \log(y_i) + (1-x_i) \log(1-y_i))" /></a>

# Variational AutoEncoder

## Description

>モデル分布の尤度が最大となるパラメタを計算する．VAEは，尤度を観測変数と潜在変数の２つに分解して計算する．  
>観測変数ｘは画像など実際に観測可能な確率変数で，潜在変数ｚは直接観測できない確率変数になる．  
>VAEは潜在変数ｚを導入るすることで，モデル分布を２つの確率変数に分解することができる．  
>オートエンコーダでは，画像の特徴量を潜在変数ｚに圧縮し，デコーダで潜在変数を画像ｘに復元した．  
>この時，潜在変数ｚはどのような分布になるか不明．  
>画像をエンコーダに通した結果の潜在変数がわからなければ，デコーダにどのような変数を入力すると画像を生成できるかも不明．  
>VAEはエンコーダの潜在変数ｚを標準正規分布に従う確率変数でモデル化する．  
>学習に使用する入力画像の特徴量を標準正規分布に押し込むことができ，VAEは標準正規分布をデコーダに入力した画像を生成する．  
>VAEの潜在変数は標準正規分布に従うようにモデル化されているので，様々な画像に対する潜在変数が潜在空間内で密集している．  

### Loss

VAEの損失関数は再構成誤差に加えて，　
潜在変数が標準正規分布に従うよう，正則化の誤差を考慮する必用がある．  
よって，

<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;J^{REC}&space;&plus;&space;J^{REG}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J&space;=&space;J^{REC}&space;&plus;&space;J^{REG}" title="J = J^{REC} + J^{REG}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=J^{REG}&space;=&space;-\frac{1}{N}&space;\sum_{i}^{N}(x_i&space;\log(y_i)&space;&plus;&space;(1-x_i)&space;\log(1-y_i))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J^{REG}&space;=&space;-\frac{1}{N}&space;\sum_{i}^{N}(x_i&space;\log(y_i)&space;&plus;&space;(1-x_i)&space;\log(1-y_i))" title="J^{REG} = -\frac{1}{N} \sum_{i}^{N}(x_i \log(y_i) + (1-x_i) \log(1-y_i))" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=J^{REG}&space;=&space;-&space;\frac{1}{2}&space;\sum_{j=1}^{J}(1&space;&plus;&space;\log(\sigma_{j}^2)&space;-&space;\mu_{j}^2&space;-&space;\sigma{j}^2)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J^{REG}&space;=&space;-&space;\frac{1}{2}&space;\sum_{j=1}^{J}(1&space;&plus;&space;\log(\sigma_{j}^2)&space;-&space;\mu_{j}^2&space;-&space;\sigma{j}^2)" title="J^{REG} = - \frac{1}{2} \sum_{j=1}^{J}(1 + \log(\sigma_{j}^2) - \mu_{j}^2 - \sigma{j}^2)" /></a>

上記のようなLossになる．  
Jregはsigma=0,mu=1のときに最小化する．  
エンコーダが出力する潜在変数ｚは標準正規分布と一致すると損失が０になり，平均と分散が標準正規分布から外れると損失が発生する．  

VAEは，ぼやける理由は潜在変数の分布や標準正規分布に従うよう制約を課し，ピクセル単位で損失を計算することが原因．  



# Reference
- https://github.com/ayukat1016/gan_sample
