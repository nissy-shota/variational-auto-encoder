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

### parctical GAN

生成モデルのシンプルなアイディア   
変換したいものに関する判断結果から初めて，変換システムのもう一つの端に画像を生成する.  
推定値であるｚを考えそこからサンプルｘ*を生成する．理想的にはx*は本物のサンプルであるxと同じくらいリアルなものが生成されることが望まれる．
この推定値ｚはいわゆる潜在空間の中にあり，これはサンプルを生成していくための：啓示：の・ようなものでいつも全く同じx*を出力することはないようにする．*  
この潜在空間は学習の結果得られた表現で，この表現を解きほぐすことで，人間にとって意味があるモデあることが望まれる．  
同じでたでも，違うモデルをつかうと違う潜在表現を学習する  
潜在空間とはデータがより簡潔に表現された，隠された空間である．  
これはｚと表現され，完結とは単に次元が低いという意味である．  
データが良い潜在表現になっていれば，その空間内で似ていてるものをグルーピングすることができる．  

AutoEncoderとGANの違い
  オートエンコーダは一つの損失関数を使って入力から出力まで最適化するが，GANは生成器と識別器にそれぞれ異なった損失関数を割当てる．  
 
AutoEncoder  
エンコーダネットワークを使って，最初のデータ表現，例えば画像xを入力として，その次の次元ｙ^からz^に減らす．  
潜在空間(z)ネットワークを訓練するにあたって，潜在空間に何かしらの意味が形成されることを目指す．  
通常は入力より小さな次元で中間ステップとして動作する．  
この表現形式はオートエンコーダがその考えをまとめたものとして捉えることができる．  
デコーダネットワーク  
デコーダは，元の表現と同じものを元の次元で再構成する．デコーダはエンコーダを鏡で移したような，構造のNNとして構成することが多い．  
このステップによってzはxに変換される．個々ではエンコーダと逆のことをする．  

学習方法  
1. 画像ｘをとりだして，オートエンコーダに入力する．  
2. ｘ^が出力される．これは再構成された画像．  
3. 再構成誤差を計算する．これはxとx^の差である．これは，xとx^の画素間の距離．例えばMSEとして計算される．これにより目地的な目的関数が定義され，勾配降下法により，最適化できるようになる．エンコーダとデコーダのpためたを再急降下法を使用して更新し，再構築時の損失を最小化する．  

オートエンコーダの利点  
自動的に圧縮表現を得ることができる．(非可逆圧縮ではない)  
次元削減された潜在空間であれば，対象となるクラスとの差分を高速に計測することができる．データ感の距離を高速に計算することができれば情報計算にも用いることができる．  
ノイズ除去や白黒画像の彩色．古くて非常にノイズの多い画像や動画からノイズを取り除き，色を深くする．  
いくつかのGANアーキテクチャは内部にAEをふくむことで訓練を安定化させる．  
AEの訓練には，ラベル付された訓練データは必要ない．  
教師なし学習である．（自己訓練）  
AEを用いて新しい画像を生成することができる．AEは文字や顔画像に適用されたが，解像度が上がってくると性能が悪くなる．  

教師なし学習について  
教師なし学習は，データが何を意味するかに関する追加のラベルなしに，データそのものから学習できる機械学習の１種である．  

変分オートエンコーダ  
変分オートエンコーダは潜在空間は単なる数の集まりではなく，学習された平均値と標準偏差を持つ，正規分布として表現される．通常は，他変量ガウス関数を使う．  
実装上は普通のオートエンコーダは潜在空間を配列として学習しようとするが，ベイズ的なオートエンコーダは分布を定義する適切なパラメタを見つけようとする．  
学習した潜在空間から値をサンプリングすることで具体的な値をえる．これをデコーダに入力することで，元のデータセットに似ているが，モデルにより新しく得られたデータが出力される．  

KL divergenceは２つの分布の交差エントロピと自己エントロピとの差．  
単峰ではなく，単純な２峰性の分布を仮定すると，結果として得られるモデルは，全くおかしなものになる（モード崩壊）  
KL的な尺度を使った場合VAEや初期のGANではよく起こる．  
真の分布を山が一つの単純なガウス分布だと仮定してしまったとすると，平均と分散を出そうとする．最尤推定法を用いて分布が単峰性であると仮定した場合間違ったことが起こる．  
仮定したモデルが間違っているので，推定される分布は，この正しい２つの分布の平均に中心を持ち，その周囲に広がる正規分布になってしまう．これを点推定と呼ぶ．  
最尤推定は２つの際立ったピークがある分布を捉えきれない手法なので，誤差を最小化しても，点推定された中心の魔w理に広く広がった分布を返してしまう．  
以上のように，点推定では，間違うことがあり，真の分布からのsンプルデータが存在しないようなところを推定してしまうことがある．  
平均だと推定したところから実際のサンプルは発生しない．  
ガウス潜在空間ｚによって何が可能になるか  
VAEはそれが見たデータをガウス関数の範囲内で表現しようとする．  
しかしガウス関数は確率質量の99.7%を3σ範囲内に収めてしまう．そのためにもVAEも安定な中心を選んでしまう．  
VAEはある意味ガウス関数の上に直接構築されているが，VAEはGANのようにシナリオまで対処することはできない．  

VAEが安定な中心を選ぶことの例  
CelebAのデータセットに対して，目や口などの必ず現れる特徴は表現されているが，背景画像はおかしくなっている．  
VAEによって生成された画像は顔の中心は揃えられていて，目や口の周りの特徴位置も揃えられているが，背景はそれぞれ異なる傾向がある．  
VAEは安全策をとって，背景をぼやかすことで損失を小さくしているが，良い画像を生成しているわけではない．  

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
