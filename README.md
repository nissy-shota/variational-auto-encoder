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

>教師なし学習の一つ
>データを表現する特徴を獲得するためのNN
>入力Xから潜在変数Zに変換するニューラルネットワークをEncoderという．
>ｚの次元が入力Xより小さい場合次元削減とみなせる．
>潜在変数ｚをインプットとして元画像を復元するNNをDecoderという．
>

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

>VAEはAEの潜在変数ｚに確率分布z~N(0,1)を改定している点が異なる．  
>AEは潜在変数Zにデータを押し込めているものの，その構造についてはよくわからない．  
>そこで，VAEは潜在変数ｚを確率分布という構造に押し込めることを可能にする．  
>VAEは正規分布に従うように設計されているため，正規分布に従う乱数が学習時に使用される．  
>この乱数による僅かな違いが，似た形状のものを知覚に寄せる効果がある．  
>つまり，同じ画像を入力しても毎回ｚがズレた位置にプロットされる．（ｚからDecoderによって生成する画像を入力画像と同じようにするため）  

>生成モデルの目的は，データの分布であるp(x)を推定することである．  
>高次元のうち，データが存在する箇所は非常に限られるので，それをうまくキャプチャして低次元の因子の潜在変数ｚで，表現することを考える．  
>高次元データXと低次元データｚの対応かんけいを構築．  
>VAEは潜在変数ｚが正規分布として分布するように学習させて，ｐ(x)を推定する．エヴィデンスと言われる場合もある．  

>p(x)に関する最尤法を用いて最もよくXを表現するp(x)のパラメタを求めることができる．この確率分布を構成する要素の一部にNNを用いてもとめる．  
>データXから潜在変数ｚを対応付けるNNをEncoderと呼ぶ．  
>潜在変数ｚからデータXを復元するNNがDecoderと呼ばれる．  

>変分加減を最大にするようなパラメタを求めれば良い．  
>変分加減はELBOと呼ばれることがある．  

1 p(x)の尤度を最大にするNNのパラメタθ，φを最尤法にてもとめる．  
2 扱いやすいように対数尤度logp(x)を最大にするターゲットとする．  
3 そのまま，logp(X)を最大にすることは積分の扱いが困難であるため，変分加減L(X,z)を最大にして下から抑えに行くことで対数尤度を再々にするパラメタを求める，  
イェンセンの不等式[link](https://qiita.com/kenmatsu4/items/26d098a4048f84bf85fb)を利用する．  
KL divergence[link](https://qiita.com/kenmatsu4/items/c107bd51503462fb677f) の最小化と同値になる．  
NNの場合，損失を最小化するように学習するため，変分加減にマイナスを欠けたものを損失関数として最適化を行う．  

#### Reparameterization Trick
先程の構造だと，確率分布が間に入っているため，誤差逆伝播法をそれより先に適用できない．  
これを解決するために，Reparamterization Trickを使用する．  
z~N(μ(X),σ(x))を直接扱うのではなく，ε~N(0,I)にてノイズを発生させ，  
z = μ(x) + ε + σ(x)という形につなげることで，VAEを構成し，確率変数を避けている．  


# Reference

## book
- [実践GAN](https://book.mynavi.jp/ec/products/detail/id=113324)
- [深層学習 (機械学習プロフェッショナルシリーズ) ](https://www.kspub.co.jp/book/detail/1529021.html)
- [GANディープラーニング実装ハンドブック](https://www.shuwasystem.co.jp/book/9784798062297.html)
  - [sample code](https://github.com/ayukat1016/gan_sample)

## article
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html)
- [Variational Auto Encoder入門+ 教師なし学習∩deep learning∩生成モデルで特徴量作成](https://speakerdeck.com/katsunoriohnishi/variational-auto-encoderru-men)
- [Factor Analysis, Probabilistic Principal Component Analysis, Variational Inference, and Variational Autoencoder: Tutorial and Survey](https://arxiv.org/abs/2101.00734)
