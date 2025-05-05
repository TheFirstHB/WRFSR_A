

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>



**深層学習生成モデルを使用したメッシュ気象データ高解像度化**

Downscaling of meteorological gridded data using deep generative model

佐久間一世（生態環境物理学研究室）


# 1. **はじめに**

気象データの活用は農業の生産性向上に大きく寄与する。例えば、農作物の生育評価（Aoki et al., 2015; Ikegami et al., 2014）、農作業計画や判断（Shinjo et al., 2015）といった農業従事者における営農支援への貢献や、将来における気候変動，異常気象に対応した農業技術の開発(Inoue, 1997;Yonemaru, 2023)などである。

気象データは観測測器の設置やリモートセンシングにより取得され、インターネットを通じてデータが利用できるような整備が進んでいる。わが国では農研機構よりメッシュ農業気象データ（AMGSD: Ohno et al., 2016）が提供されており、水平解像度約1kmで日別・時別データが整備されている。メッシュ農業気象データは病害虫の分布、発生時期の推定（Ohoe et al., 2017; Okuda et al. 2019）、収量予測(Ito et al., 2016; Mochizuki et al., 2022)、乾燥ストレス発生リスクの評価（Nakano et al., 2015）などに活用されている。

しかし、この分解能は小規模な農地が散在する日本では不十分だといえる。AMGSDデータ1格子の面積は100haであるのに対し、都府県の1経営体当たりの平均耕地面積は2.2ha、北海道に限定した場合は30.2haである（農林水産省・北海道農政事務所, 2022）。そのため、現状の解像度では周囲の土地利用や水系による影響の分離が困難である。また、経営体においても農地が密集しているとは限らず、農業経営体が栽培管理しやすいとされている耕区の大きさはおおむね 6.6a～10a とされている（農林水産省・農業農村振興局整備部会, 2007）。したがって、耕区ごとに比較・管理が可能な水平解像度を目的とした場合、さらに高解像度のデータが必要となる。

これに対し高密度の気象観測を行うことはコストの問題がある。単純な設置コストだけでなく、保守・点検のためのコストもかかるため、僻地での維持管理が難しく、高標高域では観測所の廃止が起こっている（Hayashi et al. 2017）。

従って、メッシュ気象データを必要に応じてダウンスケーリングすることが現実的な対応策であると考える。ダウンスケーリングには力学的手法と統計的手法が存在し、計算コスト、精度の面で様々な利点と欠点がある。

 力学的手法とは、領域気象モデル（RCM）を用いて高解像度のシミュレーションを行い、高解像度の気象データを取得する手法である。RCMによるシミュレーションには、全球気象モデル（GCM）の出力データを側方大気境界条件や初期条件として利用する。代表的なRCMとしては、RAMS（Pielke et al., 1992）やWRF（Skamarock et al., 2006）が挙げられる。

RCMは、入力された地形や土地利用といった地表面特性を考慮して、これらの特性を反映した気象物理計算を行うため、高解像度の地形情報に基づいたシミュレーションは、データの精度向上に大きく寄与する。例えば、降雪量のWRFシミュレーションでは、6km以下の水平解像度のRCM出力結果が観測値と良い一致を示し、GCMで見られた過大評価や過小評価が改善されたことが報告されている（Rasmussen et al., 2011）。

さらに、RCMは目的変数だけでなく、多様な気象変数を同時に出力することが可能である。（余裕があれば何か入れる）したがって、力学的ダウンスケーリングを適用することで、観測が困難な地域においても、地形や土地利用などの物理的特性を反映した、比較的精度の高い気象データを提供することが可能となる。

しかし、空間格子数、タイムステップ数の増大により、高解像度の計算を行う場合、計算コストが膨大なものとなる。また、気象データの出力はある程度の系統誤差が存在し、それらは陸面プロセスや境界層のモデル表現が不完全なことに起因する（Tang et al., 2016）。このような気象物理モデルの不確実性による出力結果と観測値の誤差軽減には、観測データとの同化が必要であり、これも計算コストの増大をもたらす。


統計的手法は、低解像度気象データに対し、与えられた地域の観測データとの統計的な関係性に基づいて再解析する高解像度化手法である。例えば、地形を考慮して、気象データを空間的に補間する手法（PRISM）(Daly et al., 1994)がある。日本でも、数値標高データを適用することで融雪流出の100mメッシュ解析（Lu et al., 1998）気温場の50mメッシュ化（Ueyama, 2008）などが行われている。

これらの手法は回帰モデリングによる統計的ダウンスケーリングに分類される。他にも、実際の気象観測データに基づき、似たような統計特性を持つ気象データを生成する生成モデリング的手法がある。これは確率的ウェザージェネレータ法（Semenov and Barrow., 1997） （WG法）に代表される。

統計的手法は、複雑な気象物理計算を介さないため、計算コストを抑えられるという利点がある。また、観測データに基づく統計モデリングであるため、気象物理モデルと比較して実測値との誤差が小さい。

しかし、出力結果が元のデータに強く依存するという欠点があり、とくに外挿領域では推定値が不安定になることが問題となる。たとえば、PRISM法によって得られたデータは、観測値が極端に少ない地域において標高への依存度が高くなり、実態とかけ離れた値を出力することがある（Gutman et al., 2012）。また、実測値は環境の変化に応じて変動するが、周期的・非周期的を問わず、こうした長期的な変化を捉える統計モデルの構築には、10年単位の継続的なデータ取得が必要となる。

長期間のデータ観測の必要なしに環境変化に応じた条件設定、シミュレーションを即座に行うことができる点で力学的手法による出力データは利用価値が高い。こうしたデータを、統計的モデルの学習に用いるハイブリットな手法が現在注目されている。

中でも、ニューラルネットワーク（NN）を利用した非線形回帰モデルによる統計的ダウンスケーリングは、深層学習技術の発展に合わせて特に利用が進んでいる手法であり、NWPデータに基づく短期風力発電予測(Yang et al., 2022)、（）に用いられている。RCMモデルをNNに学習させた例として、Sekiyama et al.（2023）では風速場の5㎞から1kmへのダウンスケーリングを行っており、低解像度データに存在した風向、風速誤差を改善した。

物理モデル出力データに基づく統計モデル構築は、力学的手法の物理的な信頼性を保ちながら、気象物理計算による重い計算負荷をかけずにダウンスケーリングすることができる(Walton et al., 2015)。

しかしながら、CMデータを単純に学習させた場合、気象物理モデル特有の系統誤差も学習してしまう。この問題に対処するために、深層学習による生成モデル（DGM）的手法が提案された。Huang et al.（2024）は深層学習生成モデルによる気象データ生成プロセスに観測データを合成するデータ同化を行った。この手法を取り入れることで、物理的根拠の付加と出力データの数値的信頼性の両立が期待される。


本研究では物理モデルからの出力を深層学習生成モデルの学習に用い、約3倍のダウンスケーリングを行うモデルを構築する。その学習過程ではHuang et al. （2024）の手法に基づく観測データの同化処理を行う。出力結果に対し、地形条件による変化の検証、実測データとの比較を通して本手法の有効性を検証する。

# 2. **方法**
## 2.1. **実験設定**
### 2.1.1. 使用データ

#### 2.1.1.1. 気圧データ

気圧データはUCAR/NCAR/CISL/DSSが提供するNCEP FNL Operational Global Analysis dataを使用した。これは、解像度1.0度、 鉛直34層 の全球気圧面データが格納されている。

#### 2.1.1.2. 観測データ

AMGSDをデータ同化用の観測値として使用した。なお、AMGSDはアメダス観測点のデータを統計処理したものであるが、ここでは便宜的に観測データとして扱っている。AMGSDはランダムサンプリングを行うことで可変数のポイントデータを作成した。期間は2021～2023の3年間とした。

### 2.1.2. 対象領域

対象地域は北海道岩見沢市周辺の地域である。この地域は平地に加え中山間地、山間部にも畑作地帯が分布している。畑作地帯の特性に応じて、岩見沢市・栗山町・栗沢町の3領域を設定した。また、この地域では、岩見沢市農業気象サービスとして地上観測が行われている。このデータを検証用のデータとして利用し、気温（K）・相対湿度（％）、の時別データを取得した。これらはメッシュ農業気象データの算出には利用していない。設定領域とアメダス・岩見沢市農業気象データの観測点を図１に示す。

**表 1：領域内の観測点の名称と位置**


![alt text](t_obspoint.png)


![alt text](f_area.png)


**図 ：対象領域
（岩見沢;N43.2329,E141.7548、栗山;N43.0905,E141.7781、栗沢;N43.1603,E141.9335
を中心とした約15kmｘ15kmの領域．黄色の点は表1の観測点の位置を示す．）**

## 2.2. **気象物理モデル**
### 2.2.1. Weather Research and Forecasting model
気象物理モデルとしてWRF（Skamarock et al. 2019）を使用した。WRFは数値気象モデルの一つであり、大気現象のシミュレーションと予測に使用される。また、運動方程式に鉛直方向の加速度を加味しているため、大気の上昇や下降の影響が大きいメソ〜領域スケールに特化したモデルである。WRFは気象予測の分野だけでなく都市気候研究（Kitao et al., 2010）、災害解析（Chawla et al., 2018）、複雑地形下での風況予測（Uchida et al., 2013; Solbakken et al., 2021）など様々な領域気象研究に用いられており、領域スケールに影響する雲、乱流、地表面を考慮する様々な物理スキームを導入することができる。


## 2.3. **深層学習生成モデル**

### 2.3.1. 確率的ノイズ除去拡散モデル

画像生成モデルの一種である Denoising Diffusion Probabilistic Models（DDPM, Ho et al. 2020）を基本構造とするモデルを使用した。要レビュー

### 2.3.2. アルゴリズム

DDPMモデルは学習プロセスと画像生成プロセスに分けられる。学習プロセスには元画像*x0*にタイムステップ*t*に応じたノイズを加えた画像*xt*が用いられる：

$$\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}$$

ここで、*αt*は加えるノイズの大きさ、*t*はノイズ強度（0≦*t*≦*T*）、*x0*は元画像、*ｘT*は平均0、分散1の正規分布に従う完全なノイズ画像である。ニューラルネットワークはすべての*t*において与えられたノイズ成分を学習し、以下に定義された損失関数を最小化するようにパラメータ*θ*を更新する：

$$L=\mathbb{E}_q\Biggl\lbrack\sum_{t=1}^T ‖ϵ_θ (x_t,t)-ϵ_t ‖^2\Biggr\rbrack$$

$$p\left(\mathbf{x}_{t-1}\mid\mathbf{x}_t\right)=\mathcal{N}\left(\mathbf{x}_{t-1};\mu_\theta\left(\mathbf{x}_t,t\right),\sigma_t^2\mathbf{I}\right)$$

は付加されたノイズ成分、が予測されたノイズ成分である。学習したモデルは確率分布を推定し、*xt*から*xt−1*の画像を生成するノイズ除去モデルとして運用される：
$$\mathbf{x}_{t-1}=p(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$$
はモデルによって推定される平均、は*ｔ*によって決まる分散（0≦≦1）である。画像生成プロセスでは、このノイズ除去モデルの階層的な適用によって最終的に完全なノイズ画像から元の画像を再構築する。

### 2.3.3. 条件設定
本研究では*t*を1000に設定し、ノイズ除去モデルにWRF高解像度データを学習させた。さらに入力データに低解像度画像を条件付けする改良を施し、WRFデータ高解像度化モデルを作成した。

##  2.4 **実装**
### 2.4.1　WRFデータ出力

#### 2.4.1.2 条件設定
図１で設定した各領域に対し、WRFデータを出力した。
メッシュサイズを1：3、外端のスケールを1000mに設定した。
物理スキームにはWRFに適用されている基本的な設定を利用したが、計算効率と精度を可能な限り両立するために表2の計算条件を設定した。
![alt text](t_WRFconfig.png)

#### 2.4.1.3 データ取得
出力された時別気象データから2m気温（K）・相対湿度（％）を利用した。最終的に、対象領域に対応する1km、333m 解像度の時別メッシュデータを計 26280 枚取得した。

**表 2：WRFの計算設定と使用データ、および導入した物理スキーム**

### 2.4.2 高解像度化モデルの実装と学習
#### 2.4.2.1 データ前処理
図1の実験領域に対応するWRF計算結果とメッシュ農業気象データをそれぞれ切り出してデータセットを作成した。これら3種類（WRF\_1km,WRF\_333m,AMGSD）のデータに対しバイキュービック補間によってメッシュサイズを揃え（64x64）、値はWRF\_333mの平均値と標準偏差に基づくZスコア標準化を行った。データ数を8:2 に分割し、それぞれ学習用、テスト用とした。

#### 2.4.2.2 メッシュ気象データ高解像度化
本モデルの全体構造を図2に示す。観測データ同化はHuang et al.（2024）の手法を使用した。具体的には、ポイントデータ座標に基づき展開し、ガウシアンフィルタ（11x11,σ＝2.0）によって平滑化した。このデータ（$X^{data}$）に対し、式（1）によるt-1ノイズ付加を行い（$X^{data}_{t-1}$）、DDPMノイズ除去データ$X{t-1}$と合成した（$X^{out}_{t-1}$）。合成されたtはt=1を満たすまで再度ノイズ除去プロセスに適用され、最終的に生成された$X^{out}_{0}$を出力データ（Model）とした。

#### 2.4.2.3 学習設定
すべての実装は python によって行い、深層学習ライブラリとして Pytorch、DDPM モデルパイプラインの構成に Diffusers ライブラリを使用した。DDPMの学習に際して、バッチサイズ 32、学習回数 10 回、学習率 0.001 と設定し、損失関数はL2Loss、最適化関数はAdamを使用した。

![alt text](f_process.png)

**図 2：モデル全体のプロセス．本モデルは（a）ポイントデータ前処理と、（b）深層学習モデルによる WRF 高解像度データ生成プロセス（図2b）と、（c）データ合成プロセスの三つで構成される．**

## 2.5. 実験環境
実験に使用したPCに搭載されているCPUはRyzen 5995WX、GPUはNVIDIA Geforce RTX 3080 Ti（VRAM：12GB）である。

#  3. **結果と考察**
## 3.1. WRF出力データ



##  3.2. **高解像度化モデルの出力**

図3a,bは実際に高解像度化を行った出力結果の一例である。気温の出力をみると（図3a）、WRFデータはAMGSDに比べ全体で2.0K程度の過大評価が生じていたのに対し、Modelではこの傾向が改善されAMGSDに近い値を出力した。また、ModelにはWRFデータに存在する特徴を反映した。特に土地被覆の影響が顕著であり、岩見沢の河川や栗沢のダム湖に対応する部分で特徴的な値の変化が見られた。相対湿度のWRF出力結果（図3b）には大気の運動に起因する値分布が強く見られたが、Modelにはこの特徴が反映されなかった。

![alt text](f_output.png)


**図 3：領域別（岩見沢,栗沢,栗山）の高解像度データ出力の結果.（a）は2022/9/23/12時の気温分布、（b）は2022年3月6日8時の相対湿度分布を示す.**

図4に同化データ数によるデータ分布の変化を示す。全体の傾向として、データ数が多くなるにつれてAMGSDの分布に近づく傾向が見られた。逆に同化データ数が少ないと不安定な出力結果が得られた。同化データ数240以下の場合、外れ値が生じており極端なデータが生成された。また、同化データ数280のようにデータ数が多くても出力が不安定な結果になることがあった。これはランダムサンプリング位置の偏りや生成モデル出力の不安定さから生じているものであると考えられる。同化データ数の増加により安定な出力を得ることは可能だが、これはWRFデータの特徴量の消失に繋がるため、適切な同化データ数の決定は今後の検討課題である。

![alt text](f_obsnum.png)


**図 4：観測データ同化数による出力結果の値分布変化（気温場の出力データ1バッチから作成．amdとwrfは対応するAMGSD、WRF\_333mデータから取得．）**

## 3.3. **WRF特徴量の検証**

出力結果に含まれるWRF特徴量を定量化するために、正規化相関法によってWRF\_300mとの構造類似度をAMGSD、出力結果に対し算出し、構造類似度を評価した。その結果、全てのケースにおいてAMGSDに比べ相関係数の増大を確認した。（図5）。このことから、出力結果にはWRFの傾向が反映されていると考えられる。一方、相対湿度における相関係数の向上は僅かであった。相対湿度は大気の運動や雲の発生の寄与が大きいため、不確実性の高い特徴量は平均化して学習していることが考えられる。WRFとの構造類似度を相関係数で取得した結果を図3に示す。全てのケースにおいてAMGSDに比べ相関係数の増大を確認した。

**図5：領域別データの構造的類似度（a:気温,b:相対湿度）**

![alt text](hist_WRFstructure.png)

## 3.4. **農地サブピクセルの検証**

気温データの各領域に対し、農地（畑、水田）に該当するピクセルのみを抜き出し、その分布を調べた。土地被覆データは宇宙航空研究開発機構（JAXA）が提供する2022年高解像度土地利用土地被覆図を利用し、領域に合わせ整形した。取得した農地ピクセルデータから時系列ごとに階級数30のヒストグラムを作成し、その形状を評価した（表4）。その結果、全ての領域において分散の増大、尖度の低下が確認された。また、ピーク数が減少していることから、AMGSDがModelに比べ離散的な値の分布をしていることが示唆される。これらから、Modelは値を大きく変えることなく分布を平滑化し、農地サブピクセルにおける推定値が分離されたといえる。

**表 4：農地サブピクセルのヒストグラム特性分析（全テストデータの平均で算出）**

![alt text](t_subpixel.png)

# 4. **まとめ**

深層学習生成モデルを利用したWRF気象データの学習を行い、生成データとメッシュ気象データを合成する高解像度データを出力するモデルを開発した。出力結果にはメッシュ農業気象データに存在しない土地的特徴を加味したデータが出力されており、これはWRFデータ由来のものである。一方、大気の運動は生成過程において大きく平均化されているため、その影響の定量化が今後の課題である。

**参考文献**

Abatzoglou JT, 2013: Development of gridded surface meteorological data for ecological applications and modelling. International journal of climatology 33, 121–131
Chen D, Qi X, Zheng Y et al., 2024: Synthetic data augmentation by diffusion probabilistic models to enhance weed recognition. Computers and Electronics in Agriculture 216, 108517
Dimet FXL, Talagrand O, 1986: Variational algorithms for analysis and assimilation of meteorological observations: theoretical aspects. Tellus A 38A, 97-110
Evensen G, 1994: Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. Journal of Geophysical Research: Oceans 99, 10143-10162
Funk C, Peterson P, Landsfeld M et al., 2015: The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. Scientific Data 2, 150066
Ho J, Jain A, Abbeel P et al., 2020: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851
Haylock MR, Hofstra N, Klein Tank AMG et al., 2008: A European daily high-resolution gridded data set of surface temperature and precipitation for 1950–2006. Journal of Geophysical Research: Atmospheres 113, 
Huang L, Gianinazzi L, Yu Y et al., 2024: DiffDA: a Diffusion Model for Weather-scale Data Assimilation. 
Ito A, Suzuki S, 2016: メッシュ農業気象データを用いたイチゴの栽培管理支援ソフトウェアの開発. 東北農業試験研究協議会 , 91−92
Kitao N, Moriyama M, Tanaka T et al., 2010: メソ気象モデル WRF を用いた大阪地域のヒートアイランド現象に関する研究 潜在自然植生の概念を用いた都市化の影響評価. 日本建築学会環境系論文集 75, 465–471
Kominami Y, Hirota K, Inoue S et al., 2015: メッシュ農業気象データのための積雪水量推定モデル. 雪氷 77, 233-246
Kusaka H, 2011: 領域気象モデル WRF の都市気候研究への応用と課題. 地学雑誌 120, 285–295
Kalman RE, 1960: A new approach to linear filtering and prediction problems. ASME. J. Basic Eng 82, 35-45
Kita M, Kawahara Y, Tsubaki R et al., 2016: WRF による 2014 年 8 月広島豪雨の数値解析. 土木学会論文集 B1 (水工学) 72, I_211–I_216
Khanna S, Liu P, Zhou L et al., 2024: DiffusionSat: A Generative Foundation Model for Satellite Imagery. 10.48550/arXiv.2312.03606
Miyoshi T, Honda Y, 2007: 気象学におけるデータ同化. 天気 54, 15–18
Nakano S, Ohno H, Shimada S et al., 2017: 発育予測モデルとメッシュ気象データを利用したダイズの乾燥ストレス発生リスクの広域評価. 生物と気象 17, 55-63
Okuda M, Hirae M, Shiba T et al., 2019: メッシュ農業気象データを利用したヒメトビウンカ発生時期の推定. 関東東山病害虫研究会報 2019, 52-55
Ohoe T, Takagi T, Yokobori A et al., 2017: 宮城県におけるクモヘリカメムシのメッシュ農業気象データを用いた分布地域の推定. 北日本病害虫研究会報 2017, 247-252
Ohno H, Sasaki K, Ohara G et al., 2016: Development of grid square air temperature and precipitation data compiled from observed, forecasted, and climatic normal data. Climate in Biosphere 16, 71-79
Powers JG, Klemp JB, Skamarock WC et al., 2017: The Weather Research and Forecasting Model: Overview, System Efforts, and Future Directions. Bulletin of the American Meteorological Society 98, 1717-1737
Rabier F, 2005: Overview of global data assimilation developments in numerical weather-prediction centres. Quarterly Journal of the Royal Meteorological Society 131, 3215-3233
Reichle RH, 2008: Data assimilation methods in the Earth sciences. Advances in Water Resources 31, 1411-1418
Rodell M, Houser PR, Jambor U et al., 2004: The Global Land Data Assimilation System. Bulletin of the American Meteorological Society 85, 381-394
Sekiyama TT, 2020: Statistical Downscaling of Temperature Distributions from the Synoptic Scale to the Mesoscale Using Deep Convolutional Neural Networks. 10.48550/arXiv.2007.10839
Sekiyama TT, Hayashi S, Kaneko R et al., 2023: Surrogate Downscaling of Mesoscale Wind Fields Using Ensemble Superresolution Convolutional Neural Networks. Artificial Intelligence for the Earth Systems 2, 
Sanaeifar A, Guindo ML, Bakhshipour A et al., 2023: Advancing precision agriculture: The potential of deep learning for cereal plant head detection. Computers and Electronics in Agriculture 209, 107875
Schultz MG, Betancourt C, Gong B et al., 2021: Can deep learning beat numerical weather prediction?. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 379, 20200097
Sun J, Xue M, Wilson JW et al., 2014: Use of NWP for nowcasting convective precipitation: Recent progress and challenges. Bulletin of the American Meteorological Society 95, 409–426
Toker A, Eisenberger M, Cremers D et al., 2024: SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation. 10.48550/arXiv.2403.16605
Uchida K, Tatumi K, Kawashima T et al., 2013: メソ気象モデル WRF-ARW を用いた複雑地形上の数値風況予測. Research Institute for Applied Mechanics, Kyushu University 144, 41-47
Wu T, Maruyama T, Zhao Q et al., 2023: Learning Controllable Adaptive Simulation for Multi-resolution Physics. 10.48550/arXiv.2305.01122
Yusuke A, Ryuzo O, Hideki K et al., 2014: 力学的ダウンスケーリングによる近未来標準気象データ作成に関する研究. 生産研究 66, 61-68
Yuan Q, Shen H, Li T et al., 2020: Deep learning in environmental remote sensing: Achievements and challenges. Remote Sensing of Environment 241, 111716
Zhang X, Tian S, Wang G et al., 2023: DiffUCD:Unsupervised Hyperspectral Image Change Detection with Semantic Correlation Diffusion Model. 10.48550/arXiv.2305.12410

