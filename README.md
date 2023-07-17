# LLMとLoRAを用いたテキスト分類

大規模言語モデル(LLM)は昨今ますます注目を集めていますが、zero/few-shot学習能力を評価されることが多く、BERTなど既存の事前学習済みモデルのようにfine-tuningを行ってテキスト分類をさせる、という用途にはそこまで利用されていないような気がしています。
そこで、LLMはどのくらいテキスト分類ができるのか調べるため、BERTを用いたテキスト分類と同様の方法によって、LLMをテキスト分類に用いる実験を行いました。

## モデル概要

本実験の目的は、「zero/few-shot学習能力が注目されがちなLLMを、通常のテキスト分類に用いた場合にどうなるか」について調べることです。

今までテキスト分類によく利用されていたBERTは双方向のモデルであり、テキスト分類のために文頭トークン`[CLS]`をよく利用していました。
しかし、最近よく利用されるLLM、例えばLLaMAなどは単方向のモデルです。
そのため、単方向のモデルでは文頭トークンを取ることに意味がありません。
そこで本実装では、`transformers`の[`LlamaForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaForSequenceClassification)クラスを参考に、文末トークンの埋め込み表現をテキスト分類に利用します。
単方向言語モデルにおける文末トークンは、系列中で唯一文全体のトークンを考慮可能なので、`[CLS]`の代替として適切であると考えられます。

また、LLMをFull Fine-tuningするのはメモリ・計算効率的な観点から非常に大変なので、追加の低ランク行列のみを調整することで、Full Fine-tuningと同等の性能を達成できる微調整手法であるLoRAを利用します。
備考: [LoRAの解説資料](https://speakerdeck.com/hpprc/lun-jiang-zi-liao-lora-low-rank-adaptation-of-large-language-models)
LoRAによる微調整のため、[PEFT](https://github.com/huggingface/peft)を利用します。


## 評価実験

評価実験では、livedoorニュースコーパスの9値分類を行います。
実験内容は、筆者の[BERTによるテキスト分類チュートリアル](https://github.com/hppRC/bert-classification-tutorial)とほぼ同様です。

評価実験では、7種類の日本語LLMを用いました。
具体的には、rinna社の3.6Bモデル4種類と、CyberAgent社の7B, 3B, 1Bモデルについてそれぞれ実験を行いました。

ハイパーパラメータの調整として、学習率を1e-4, 3e-4, 5e-4, 1e-3に設定してそれぞれ実験を行いました。
また、モデルへの入力の形式を3種類実験しました。
具体的には、ライブドアニュースコーパス中の各記事について、タイトルを`title`, 記事本文を`body`という変数に格納し、以下の3つのテンプレートに注入しました。

| Template Type |                                       見た目 |
| :-----------: | -------------------------------------------: |
|       0       | f"タイトル: {title}\n本文: {body}\nラベル: " |
|       1       |           f"タイトル: {title}\n本文: {body}" |
|       2       |                           f"{title}\n{body}" |

以上の、学習率・テンプレートについてすべての組み合わせで1回ずつ実験を行い、開発セットでのmacro平均F値が最も高くなったハイパーパラメータを最終的なテストセットでの評価に用いました。
LoRAのランクrは32に固定しています。

実験結果に対する注意ですが、実験は単一の乱数シード値で1度しか実施しておらず、分割交差検証も行っていないので、実験結果の正確性は高くありません。
したがって、以下の結果は過度に信用せず、参考程度に見てもらうよう、お願いいたします。

では、結果を以下の表に示します。
実験結果は、macro平均F値について降順に並んでいます。
なお、以降の結果はすべて`results`ディレクトリに保存されているCSVファイルから確認することができます。

|                                                                                                                           | Accuracy  | Precision | Recall |    F1     |
| ------------------------------------------------------------------------------------------------------------------------- | :-------: | :-------: | :----: | :-------: |
| [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2) | **97.96** |   97.77   | 97.76  | **97.75** |
| [rinna/japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)                                       |   97.55   |   97.24   | 97.39  |   97.30   |
| [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft)       |   97.55   |   97.32   | 97.27  |   97.27   |
| [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)       |   97.55   |   97.03   | 97.37  |   97.18   |
|                                                                                                                           |           |           |        |           |
| [cyberagent/open-calm-7b](https://huggingface.co/cyberagent/open-calm-7b)                                                 |   97.01   |   96.76   | 96.42  |   96.55   |
| [cyberagent/open-calm-3b](https://huggingface.co/cyberagent/open-calm-3b)                                                 |   96.88   |   96.38   | 96.51  |   96.42   |
| [cyberagent/open-calm-1b](https://huggingface.co/cyberagent/open-calm-1b)                                                 |   94.43   |   94.24   | 93.80  |   93.98   |

表から、指示チューニングされた`rinna/japanese-gpt-neox-3.6b-instruction-sft-v2`が最も高いF値を示したことがわかります。
一方で、7Bと比較的大きなモデルである`cyberagent/open-calm-7b`は若干低めのF値となりました。
より性能を高めるためには、RoLAのrやその他のハイパラなど、もうすこしチューニングしてあげる必要があるのかもしれません。

ちなみに、`rinna/japanese-gpt-neox-3.6b-instruction-sft-v2`のF値`97.75`は、筆者の別実装、[BERTによるテキスト分類チュートリアル](https://github.com/hppRC/bert-classification-tutorial)における最高性能を達成した`studio-ousia/luke-japanese-large-lite`のF値`97.47`よりも高い結果です。
もちろん、モデルのパラメータ数が9倍ほど違うので単純な比較対象にはなり得ませんが、テキスト分類の性能を追い求めたい場合には、BERTの代替としてLLM+LoRAを利用するのもよい選択肢になるかもしません。


次に、今回の実験で代表的な3つのモデル`rinna/japanese-gpt-neox-3.6b-instruction-sft-v2`, `rinna/japanese-gpt-neox-3.6b`, `cyberagent/open-calm-7b`についての、テンプレートごとの実験結果を以下の表に示します。

|                                                                                                                           | Template | Val. F1 |  F1   |
| ------------------------------------------------------------------------------------------------------------------------- | :------: | :-----: | :---: |
| [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2) |    2     |  97.27  | 97.75 |
| [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2) |    1     |  97.18  | 97.14 |
| [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2) |    0     |  97.05  | 96.80 |
|                                                                                                                           |          |         |       |
| [rinna/japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)                                       |    1     |  97.14  | 97.30 |
| [rinna/japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)                                       |    2     |  96.92  | 97.36 |
| [rinna/japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)                                       |    0     |  96.61  | 96.69 |
|                                                                                                                           |          |         |       |
| [cyberagent/open-calm-7b](https://huggingface.co/cyberagent/open-calm-7b)                                                 |    1     |  97.22  | 96.55 |
| [cyberagent/open-calm-7b](https://huggingface.co/cyberagent/open-calm-7b)                                                 |    0     |  97.07  | 96.56 |
| [cyberagent/open-calm-7b](https://huggingface.co/cyberagent/open-calm-7b)                                                 |    2     |  96.88  | 96.85 |

一般的に、LLMの推論能力はテンプレート(プロンプト)によって大きく左右されます。
一方で今回の実験は、zero/few-shot的な設定ではないので、ある程度テンプレートによる性能差を緩和できると予想されます。
しかし、結果から、テンプレートによって以前としてF値にある程度(F値にして1ポイント程度)の差が出ていることがわかります。
`template_type=0`は比較的複雑なテンプレートで、`template_type=2`は改行で連結しているだけのシンプルなテンプレートになりますが、意外と`template_type=2`のような簡単なものの方が性能が高い傾向にあることが伺えます。
zero/few-shot設定ではプロンプトが非常に重要になりますが、微調整を行える場合には、プロンプトはできるだけシンプルに済ませた方がいいということなのかもしれません。


次に、モデルを`rinna/japanese-gpt-neox-3.6b`、`template_type`を`2`に固定した場合の、学習率ごとの性能を見てみます。

|  LR   |  Val. F1  | Accuracy | Precision | Recall |    F1     |
| :---: | :-------: | :------: | :-------: | :----: | :-------: |
| 5e-2  |   2.18    |  12.91   |   1.43    | 11.11  |   2.54    |
| 3e-2  |   2.18    |  12.91   |   1.43    | 11.11  |   2.54    |
| 1e-2  |   2.18    |  12.91   |   1.43    | 11.11  |   2.54    |
| 5e-3  |   24.78   |  32.20   |   36.30   | 30.27  |   28.21   |
| 3e-3  |   2.18    |  12.91   |   1.43    | 11.11  |   2.54    |
| 1e-3  | **96.92** |  97.69   |   97.51   | 97.27  | **97.36** |
| 5e-4  | **96.77** |  98.23   |   98.02   | 97.87  | **97.93** |
| 3e-4  |   96.74   |  96.88   |   96.46   | 96.21  |   96.30   |
| 1e-4  |   94.79   |  97.01   |   96.85   | 96.72  |   96.76   |
| 5e-5  |   94.28   |  95.92   |   95.73   | 95.50  |   95.58   |
| 3e-5  |   93.74   |  94.02   |   93.50   | 93.61  |   93.55   |
| 1e-5  |   78.94   |  81.25   |   80.21   | 79.43  |   79.62   |

表から、LoRAでの学習にはある程度大きな学習率が有効であるものの、その上限は`1e-3`くらいで、`1e-2`などの非常に大きな学習率を使うと、学習がうまくいかなくなってしまうことがわかります。
もう少し広範なモデルでの実験結果が欲しいところですが、LLM+LoRAで分類を行う場合は、`5e-4`くらいの学習率を初手で試すのが安牌ではないかなと思います。

最後に、モデルを`rinna/japanese-gpt-neox-3.6b`、`template_type`を`2`に固定した場合の、LoRAのrごとの性能を見てみます。

| LoRA r |  LR   |  Val. F1  | Accuracy | Precision | Recall |    F1     |
| -----: | :---: | :-------: | :------: | :-------: | :----: | :-------: |
|      8 | 5e-4  | **97.45** |  97.15   |   96.97   | 96.75  |   96.83   |
|     64 | 1e-3  |   97.22   |  97.28   |   96.96   | 96.85  |   96.89   |
|     16 | 1e-3  |   97.20   |  97.69   |   97.59   | 97.27  |   97.38   |
|      4 | 3e-4  |   97.12   |  97.69   |   97.64   | 97.24  | **97.40** |
|     32 | 1e-3  |   96.92   |  97.69   |   97.51   | 97.27  |   97.36   |

結果としては、開発セットでのF値とテストセットでのF値の間の相関があまりみられないような気がします。
LoRAのrは「大きいモデルほど小さくできる」値だと思われるので、数B程度の中規模以下のLLMでは32以上とかにしておくのが無難な気がしますが、もう少し実験してみたい結果になりました。


## まとめ

本実装では、LLMをtraditionalなテキスト分類に用いる実験を行いました。
結果として、LoRAを用いた微調整を行うことで、ごく少数のパラメータを調整するのみで、かなり高い性能を達成することができ、「BERTの代替としてLLMを利用する」のも十分reasonableな選択肢と言えそうな結果となりました。
また、微調整を行う設定でも、依然としてテンプレートが性能に影響を及ぼすという傾向が見られました。
さらに、LoRAを利用した微調整を行う場合、学習率はかなり大きめの値に設定する必要があり、ランク`r`の値によっても性能に影響がありそうだということがわかりました。


## 参考文献
- [【インターンレポート】6.7B日本語モデルに対するLoRAチューニング](https://engineering.linecorp.com/ja/blog/lora-tuning-for-japanese-model)
- [研究のためのPython開発環境](https://zenn.dev/zenizeni/books/a64578f98450c2)
- [BERTによるテキスト分類チュートリアル](https://github.com/hppRC/bert-classification-tutorial)

## 著者情報・引用

作者: [Hayato Tsukagoshi](https://hpprc.dev) \
email: [research.tsukagoshi.hayato@gmail.com](mailto:research.tsukagoshi.hayato@gmail.com)

論文等で本実装を参照する場合は、以下をお使いください。

```bibtex
@misc{
  hayato-tsukagoshi-2023-llm-lora-classification,
  title = {{Text Classification with LLMs and LoRA}},
  author = {Hayato Tsukagoshi},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hppRC/llm-lora-classification}},
  url = {https://github.com/hppRC/llm-lora-classification},
}
```
