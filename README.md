# [IT基礎教養<br>自然言語処理＆画像解析<br>"生成AI"を生み出す技術](https://book.impress.co.jp/books/1123101097)
- 書籍内の演習で使用するipynbファイルと、データが掲載されています
- 執筆時に使用した各種パッケージのバージョンは[こちら](執筆時に使用したパッケージバージョン.txt)です

### 本の情報
- 発売日：2024/10/18
- 著者：鎌形桂太　著/三好大悟　監修 
- ISBN：9784295020318

---
## 第４章
- 文章分類問題を解いてみよう
  - [chap04_movie_review_classification.ipynb](chap04_movie_review_classification.ipynb)
## 第５章
- 言語モデルを動かしてみよう
  - [chap05_LLM_intro.ipynb](./chap05_LLM_intro.ipynb)
    - ① 穴埋め問題を解くMLM
    - ② 次のトークンを予測するCLM
## 第７章
- 画像分類問題を解いてみよう
  - [chap07_mnist_digit_classification.ipynb](chap07_mnist_digit_classification.ipynb)
## 第８章
- オートエンコーダを作ってみよう
  - [chap08_mnist_digit_AutoEncoder.ipynb](chap08_mnist_digit_AutoEncoder.ipynb)
- VAEを作ってみよう
  - [chap08_mnist_digit_VAE.ipynb](chap08_mnist_digit_VAE.ipynb)

---
### 参考：本書内で紹介した各URL
- ▶をクリックすると展開できます
- []内の番号は本文の脚注番号です
<details><summary>第1章</summary>

- [1] [similarweb Blog](https://www.similarweb.com/blog/insights/ai-news/chatgpt-birthday/)
- [3] [similarweb社による上位ウェブサイトランキング](https://www.similarweb.com/ja/top-websites/), [ChatGPT への月間アクセス数](https://www.similarweb.com/ja/website/chatgpt.com/)
- [5] [自治体AI zevo](https://prtimes.jp/main/html/rd/p/000000085.000056138.html)
- [6] [埼玉県戸田市によるChatGPTに関する調査研究](https://www.city.toda.saitama.jp/uploaded/attachment/62855.pdf)
- [7] [ディープフェイク（deepfake）を用いて作成されたCM動画](https://www.youtube.com/watch?v=XSUQwwOm3G4)
- [8] [Bruce Willis denies selling rights to his face](https://www.bbc.com/news/technology-63106024)
- [12] [Hugging Face](https://huggingface.co/), [Hugging Face](https://huggingface.co/), [Civitai](https://civitai.com/)
- [17] [A Comprehensive Survey on Applications of Transformers for Deep Learning Tasks](https://doi.org/10.48550/arXiv.2306.07303)
- [21] [On the Opportunities and Risks of Foundation Models](https://doi.org/10.48550/arXiv.2108.07258)
- [22] [Language Models are Few-Shot Learners](https://doi.org/10.48550/arXiv.2005.14165)
- [24] [On the Opportunities and Risks of Foundation Models](https://doi.org/10.48550/arXiv.2108.07258)
- [25] [Learning Transferable Visual Models From Natural Language Supervision](https://doi.org/10.48550/arXiv.2103.00020)
</details>

<details><summary>第2章</summary>

- [1] [23-1444 - Federal Trade Commission v. Automators LLC et al](https://www.govinfo.gov/app/details/USCOURTS-casd-3_23-cv-01444)
- [2] [AI Washing](https://www.techopedia.com/ai-washing-everything-you-need-to-know/2/34841)
- [3] [令和元年版情報通信白書](https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r01/pdf/index.html)
- [4] [A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence](https://doi.org/10.1609/aimag.v27i4.1904)
- [5] [A Collection of Definitions of Intelligence](https://arxiv.org/abs/0706.3639v1)
- [6] [Introducing Superalignment](https://openai.com/blog/introducing-superalignment)
- [7] [Mark Zuckerberg’s new goal is creating artificial general intelligence](https://www.theverge.com/2024/1/18/24042354/mark-zuckerberg-meta-agi-reorg-interview)
- [8] [Rule-based Expert Systems : The MYCIN Experiments of the Stanford Heuristic Programming Project](https://doi.org/10.1016/0004-3702(85)90067-0), [Computer-Based Medical Consultations: Mycin](https://doi.org/10.1016/B978-0-444-00179-5.X5001-X)
- [12] [Siri, Siri, in my hand: Who’s the fairest in the land? On the interpretations, illustrations, and implications of artificial intelligence](https://doi.org/10.1016/j.bushor.2018.08.004)
- [16] [Does Deep Blue use Artificial Intelligence?](https://doi.org/10.3233/ICG-1997-20404)
- [17] [Recommendation of the Council on Artificial Intelligence](https://legalinstruments.oecd.org/en/instruments/oecd-legal-0449)
- [18] [H.R.6216 - National Artificial Intelligence Initiative Act of 2020, SEC. 3 (3)](https://www.congress.gov/bill/116th-congress/house-bill/6216/text#H8B1131A84B984501B54FCB9DCCF19B57)
- [20] [H.R.6216 - National Artificial Intelligence Initiative Act of 2020, SEC. 3 (9)](https://www.congress.gov/bill/116th-congress/house-bill/6216/text#HE78BB0D61F4849B2BB8A77C3046E1CF6)
- [28] [A survey on semi-supervised learning](https://doi.org/10.1007/s10994-019-05855-6)
- [29] [Unsupervised and self-supervised deep learning approaches for biomedical text mining ](https://doi.org/10.1093/bib/bbab016), [Self-supervised Learning: A Succinct Review](https://doi.org/10.1007/s11831-023-09884-2)
- [33] [Mastering the game of Go with deep neural networks and tree search](http://dx.doi.org/10.1038/nature16961)
- [34] [Training language models to follow instructions with human feedback](https://doi.org/10.48550/arXiv.2203.02155), [Introducing ChatGPT](https://openai.com/blog/chatgpt)
- [39] [Updates to the OECD’s definition of an AI system explained](https://oecd.ai/en/wonk/ai-system-definition-update)
- [46] [1.1. Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#:~:text=to%20minimize%20the-,residual%20sum%20of%20squares,-between%20the%20observed)
- [64] [Visualizing the Loss Landscape of Neural Nets](https://doi.org/10.48550/arXiv.1712.09913), [Loss Visualization](https://www.telesens.co/loss-landscape-viz/viewer.html)
- [66] [A logical calculus of the ideas immanent in nervous activity](https://doi.org/10.1007/BF02478259)
- [68] [The perceptron: A probabilistic model for information storage and organization in the brain](https://doi.org/10.1037/h0042519)
- [69] [MARK I PERCEPTRON OPERATORS' MANUAL](https://apps.dtic.mil/sti/tr/pdf/AD0236965.pdf)
- [70] [Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark](https://doi.org/10.48550/arXiv.2109.14545)
</details>

<details><summary>第3章</summary>

- [1] [Is ChatGPT A Good Translator? Yes With GPT-4 As The Engine](https://doi.org/10.48550/arXiv.2301.08745)
- [2] [生成 AI による検索体験 (SGE) のご紹介](https://japan.googleblog.com/2023/08/search-sge.html)
- [3] [Introducing Duolingo Max, a learning experience powered by GPT-4](https://blog.duolingo.com/duolingo-max/)
- [10] [日本語の自然言語処理ライブラリ「GiNZA」](https://www.recruit.co.jp/newsroom/2019/0402_18331.html)
- [11] [日本語形態素解析における未知語処理の一手法―既知語から派生した表記と未知オノマトペの処理―](https://doi.org/10.5715/jnlp.21.1183)
- [14] [pneumonoultramicroscopicsilicovolcanoconiosis](https://www.oed.com/dictionary/pneumonoultramicroscopicsilicovolcanoconiosis_n)
- [20] [SentencePiece](https://github.com/google/sentencepiece)
- [21] [OpenAIのTokenizer](https://platform.openai.com/tokenizer)
- [33] [実践で学ぶBM25 - パート2：BM25のアルゴリズムと変数](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [40] [Efficient Estimation of Word Representations in Vector Space](https://doi.org/10.48550/arXiv.1301.3781)
</details>

<details><summary>第4章</summary>

- [3] [標準規格概要（STD-B10）](https://www.arib.or.jp/kikaku/kikaku_hoso/desc/std-b10.html)
- [9] [Update on Our Progress on AI and Hate Speech Detection](https://about.fb.com/news/2021/02/update-on-our-progress-on-ai-and-hate-speech-detection/)
- [10] [Updates on Comment Spam & Abuse](https://support.google.com/youtube/thread/192701791)
- [11] [深層学習を用いた自然言語処理モデル（AI）のAPIを無償提供](https://news.yahoo.co.jp/newshack/information/comment_API.html)
- [12] [日本語評価極性辞書](https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html)
- [13] [景気単語極性辞書の構築とその応用](https://doi.org/10.5715/jnlp.29.1233)
- [18] [最適な Colab のプランを選択する](https://colab.research.google.com/signup/pricing?hl=ja)
- [23] [Pythonプログラミング入門の教材](https://github.com/UTokyo-IPP/utokyo-ipp.github.io), [Chainer Tutorial](https://tutorials.chainer.org/ja/tutorial.html)
- [27] [Mersenne twister](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%83%AB%E3%82%BB%E3%83%B3%E3%83%8C%E3%83%BB%E3%83%84%E3%82%A4%E3%82%B9%E3%82%BF)
</details>

<details><summary>第5章</summary>

- [6] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.48550/arXiv.1810.04805), [Deep contextualized word representations](https://doi.org/10.48550/arXiv.1802.05365)
- [10] [bert-base-japanese-whole-word-masking](https://huggingface.co/tohoku-nlp/bert-base-japanese-whole-word-masking)
- [11] [A Primer in BERTology: What we know about how BERT works](https://doi.org/10.48550/arXiv.2002.12327)
- [12] [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://doi.org/10.48550/arXiv.1804.07461)
- [13] [JGLUE: Japanese General Language Understanding Evaluation](https://github.com/yahoojapan/JGLUE)
- [14] [GLUE leaderboard](https://gluebenchmark.com/leaderboard)
- [15] [japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium)
- [20] [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](https://doi.org/10.48550/arXiv.1610.02424)
- [21] [The Curious Case of Neural Text Degeneration](https://doi.org/10.48550/arXiv.1904.09751)
- [27] [Finetuned Language Models Are Zero-Shot Learners](https://doi.org/10.48550/arXiv.2109.01652)
- [28] [databricks-dolly-15k-ja ](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
- [32] [LLMのための日本語インストラクションデータ 公開ページ](https://liat-aip.sakura.ne.jp/wp/llm%e3%81%ae%e3%81%9f%e3%82%81%e3%81%ae%e6%97%a5%e6%9c%ac%e8%aa%9e%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%a9%e3%82%af%e3%82%b7%e3%83%a7%e3%83%b3%e3%83%87%e3%83%bc%e3%82%bf%e4%bd%9c%e6%88%90/llm%e3%81%ae%e3%81%9f%e3%82%81%e3%81%ae%e6%97%a5%e6%9c%ac%e8%aa%9e%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%a9%e3%82%af%e3%82%b7%e3%83%a7%e3%83%b3%e3%83%87%e3%83%bc%e3%82%bf-%e5%85%ac%e9%96%8b/)
- [33] [Fine tuning is for form, not facts](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts)
- [39] [AttentionViz: A Global View of Transformer Attention](https://doi.org/10.48550/arXiv.2305.03210)
- [40] [Attention Is All You Need](https://doi.org/10.48550/arXiv.1706.03762)
- [43] [Scaling Laws for Neural Language Models](https://doi.org/10.48550/arXiv.2001.08361), [Language Models are Few-Shot Learners](https://doi.org/10.48550/arXiv.2005.14165)
- [44] [Emergent Abilities of Large Language Models](https://doi.org/10.48550/arXiv.2206.07682)
- [45] [BIG-bench tasks](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/README.md)
- [49] [Are Emergent Abilities of Large Language Models a Mirage?](https://doi.org/10.48550/arXiv.2304.15004)
- [51] [Finetuning an LLM: RLHF and alternatives (Part I)](https://medium.com/mantisnlp/finetuning-an-llm-rlhf-and-alternatives-part-i-2106b95c8087)
- [54] [検索拡張生成（RAG）とは？](https://www.elastic.co/jp/what-is/retrieval-augmented-generation)
- [55] [NotebookLM](https://notebooklm.google/)
- [56] [PRtimes上で「RAG」と検索した結果](https://prtimes.jp/main/action.php?run=html&page=searchkey&search_word=RAG)
</details>

<details><summary>第6章</summary>

- [24] [図6-02-11をDeep playground上で再現](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.01&regularizationRate=0.001&noise=0&networkShape=3&seed=0.06305&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&numHiddenLayers_hide=false&percTrainData_hide=false&regularizationRate_hide=false&learningRate_hide=false&playButton_hide=false&batchSize_hide=false&problem_hide=false&noise_hide=false&activation_hide=false&stepButton_hide=false&showTestData_hide=false&dataset_hide=false&regularization_hide=false&resetButton_hide=false&discretize_hide=false)
- [25] [図6-02-12をDeep playground上で再現](https://playground.tensorflow.org/#activation=relu&regularization=L1&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.01&regularizationRate=0.001&noise=0&networkShape=8,7,6&seed=0.26458&showTestData=false&discretize=false&percTrainData=90&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&numHiddenLayers_hide=false&percTrainData_hide=false&regularizationRate_hide=false&learningRate_hide=false&playButton_hide=false&batchSize_hide=false&problem_hide=false&noise_hide=false&activation_hide=false&stepButton_hide=false&showTestData_hide=false&dataset_hide=false&regularization_hide=false&resetButton_hide=false&discretize_hide=false)
- [31] [convolution-shape-calculator](https://zimonitrome.github.io/convolution-shape-calculator/)
- [33] [Image Kernels](https://setosa.io/ev/image-kernels/), [Image-Convolution-Playground](https://generic-github-user.github.io/Image-Convolution-Playground/)
- [35] [CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization](https://doi.org/10.48550/arXiv.2004.15004)
- [36] [Gradient-based learning applied to document recognition](http://dx.doi.org/10.1109/5.726791)
- [37] [ImageNet Classification with Deep ConvolutionalNeural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [38] [MARK I PERCEPTRON OPERATORS' MANUAL](https://apps.dtic.mil/sti/tr/pdf/AD0236965.pdf)
- [39] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://doi.org/10.48550/arXiv.1409.1556)
- [40] [Comparative Analysis of Steering Angle Prediction For Automated Object Using Deep Neural Network](http://dx.doi.org/10.36227/techrxiv.16913443)
- [44] [Clinical ABCDE rule for early melanoma detection](https://doi.org/10.1684/ejd.2021.4171)
- [45] [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [47] [FixCaps: An Improved Capsules Network for Diagnosis of Skin Cancer](https://doi.org/10.1109/ACCESS.2022.3181225)
- [48] [SkinVision](https://www.skinvision.com)
- [49] [ステートメント：人工知能AIと病理医について](https://pathology.or.jp/ippan/AI-statement.html)
- [50] [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
- [51] [You Only Look Once: Unified, Real-Time Object Detection](https://doi.org/10.48550/arXiv.1506.02640)
- [52] [SSD: Single Shot MultiBox Detector](https://doi.org/10.48550/arXiv.1512.02325)
- [53] [MRI画像から神経膠腫の疑いのある領域を精密に抽出するAI技術を共同開発](https://www.fujifilm.com/jp/ja/news/list/11159)
- [54] [The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI](https://arxiv.org/abs/2405.18368)
</details>

<details><summary>第7章</summary>

- [8] [Reproducibility in Keras Models](https://keras.io/examples/keras_recipes/reproducibility_recipes/)
- [19] [2-2 郵政省における“手書き文字読取方式”の区分機](https://doi.org/10.3169/itej1954.28.257)
</details>

<details><summary>第8章</summary>

- [1] [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647)
- [3] [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [12] [Auto-Encoding Variational Bayes](https://doi.org/10.48550/arXiv.1312.6114)
- [14] [chap08_mnist_digit_VAE.ipynb](chap08_mnist_digit_VAE.ipynb)
- [15] [Generative Adversarial Networks](https://doi.org/10.48550/arXiv.1406.2661)
- [16] [This X Does Not Exist](https://thisxdoesnotexist.com/)
- [17] [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://doi.org/10.48550/arXiv.2103.17249)
- [18] [Only a Matter of Style: Age Transformation Using a Style-Based Regression Model](https://doi.org/10.48550/arXiv.2102.02754)
- [19] [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [21] [Learning Transferable Visual Models From Natural Language Supervision](https://doi.org/10.48550/arXiv.2103.00020)
- [23] [ImageNet Data](https://www.image-net.org/download.php)
- [25] [Brazil: Children’s Personal Photos Misused to Power AI Tools](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools), [YFCC100M: the new data in multimedia research](https://doi.org/10.1145/2812802)
- [26] [High-Resolution Image Synthesis with Latent Diffusion Models](https://doi.org/10.48550/arXiv.2112.10752)
- [30] [Ciditai](https://civitai.com/)
- [32] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)
- [34] [Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://doi.org/10.48550/arXiv.2301.12661), [Make-An-Audioによって生成した音声](https://text-to-audio.github.io/)
- [37] [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://doi.org/10.48550/arXiv.2211.06687)
- [38] [Introducing Stable Audio Open - An Open Source Model for Audio Samples and Sound Design](https://stability.ai/news/introducing-stable-audio-open)
- [39] [Stable Video Diffusion](https://stability.ai/stable-video)
</details>

---
更新履歴：
- 
