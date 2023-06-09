# Domain-Specific LM Papers
A compilation of papers related to domain specific language model training and evaluation. We focus on language models trained for biomedicine, finance, law, education, etc. 
   
* [0. Surveys](#0-surveys)
* [1. Domain Specific Pre-Training](#1-domain-specific-pre-training)
  * [1.1 Pre-Training from Scratch](#11-pre-training-from-scratch)
  * [1.2 Further Pre-Training](#12-further-pre-training)
  * [1.3 Mixed Pre-Training](#13-mixed-pre-training)
  * [1.4 Domain-Specific Fine-Tuning](#14-domain-specific-fine-tuning)
  * [1.5 Domain-Specific Pre-Training Objectives](#15-domain-specific-pre-training-objectives)
  * [1.6 Domain-Specific Tool Use](#16-domain-specific-tool-use)
* [2. Using Domain-Knowledge in Large Language Models](#2-using-domain-knowledge-in-large-language-models)
  * [2.1 Black-Box Retrieval Augmentation](#21-black-box-retrieval-augmentation)
  * [2.2 Retrieval Based Pre-Training](#22-retrieval-based-pre-training)
  * [2.3 Generalist and Domain-Specific Ensembles](#23-generalist-and-domain-specific-ensembles)
* [3. Miscellaneous](#3-miscellaneous)

## 0. Surveys

**Beyond One-Model-Fits-All: A Survey of Domain Specialization for Large Language Models.**
*Ling, Chen et al.* [[abs](https://arxiv.org/abs/2305.18703)], 2023

**Do We Still Need Clinical Language Models?**
*Lehman, Eric P. et al.* [[abs](https://arxiv.org/abs/2302.08091)], 2023

**Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers.**
*Tay, Yi et al.* [[abs](https://arxiv.org/abs/2109.10686)], 2021

**The Shaky Foundations of Clinical Foundation Models: A Survey of Large Language Models and Foundation Models for EMRs.**
*Wornow, Michael et al.* [[abs](https://arxiv.org/abs/2303.12961)], 2023

**A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity.**
*Longpre, S. et al.* [[abs](https://arxiv.org/abs/2305.13169)], 2023

**Towards More Robust NLP System Evaluation: Handling Missing Scores in Benchmarks.**
*Himmi, Anas et al.* [[abs](https://arxiv.org/abs/2305.10284)], 2023

**OpenAGI: When LLM Meets Domain Experts.**
*Ge, Yingqiang et al.* [[abs](https://arxiv.org/abs/2304.04370)], 2023

**Domain Mastery Benchmark: An Ever-Updating Benchmark for Evaluating Holistic Domain Knowledge of Large Language Model-A Preliminary Release.**
*Gu, Zhouhong et al.* [[abs](https://arxiv.org/abs/2304.11679)], 2023

**Adapting a Language Model While Preserving its General Knowledge.**
*Ke, Zixuan et al.*  [[abs](https://aclanthology.org/2022.emnlp-main.693.pdf)] Conference on Empirical Methods in Natural Language Processing, 2023.


## 1. Domain Specific Pre-Training


### 1.1 Pre-Training from Scratch

**Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing.**
*Gu, Yu et al.* [[doi](https://doi.org/10.1145/3388440)], 2020.

**BioMedLM: a Domain-Specific Large Language Model for Biomedical Text.**
*Venigalla, Abhinav et al.* [[doi](https://www.mosaicml.com/blog/introducing-pubmed-gpt)], 2022.

**BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model.**
*Yuan, Hongyi et al.* [[workshop](https://www.aclweb.org/anthology/2022.bionlp-1.1/)], 2022.

**SecureBERT: A Domain-Specific Language Model for Cybersecurity.**
*Aghaei, Ehsan et al.* 2022.

**LEGAL-BERT: The Muppets straight out of Law School.**
*Chalkidis, Ilias et al.* [[abs](https://arxiv.org/abs/2010.02559)], 2020

**DarkBERT: A Language Model for the Dark Side of the Internet.**
*Jin, Youngjin et al.* [[abs](https://arxiv.org/abs/2305.08596)], 2023

**A Japanese Masked Language Model for Academic Domain.**
*Yamauchi, Hiroki et al.* [[SDP](https://www.aclweb.org/anthology/2022.sdp-1.1/)], 2022.

**Galactica: A Large Language Model for Science.**
*Taylor, Ross et al.* [[abs](https://arxiv.org/abs/2211.09085)], 2022

**Language Model for Statistics Domain.**
*Jeong, Young-Seob et al.* [[doi](https://doi.org/10.1109/AI4I.2022.00022)], 2022

**SsciBERT: a pre-trained language model for social science texts.**
*Shen, Si et al.* [[doi](https://doi.org/10.1007/s11192-021-04122-5)], 2022.

**XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters.**
*Zhang, Xuanyu et al.* [[abs](https://arxiv.org/abs/2305.12002)], 2023

**Strategy to Develop a Domain-specific Pre-trained Language Model: Case of V-BERT, a Language Model for the Automotive Industry.**
*Kim, Younha et al.* [[source](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10502022)], 2023

**ITALIAN-LEGAL-BERT: A Pre-trained Transformer Language Model for Italian Law.**
*Licari, Daniele and Giovanni Comandé.* [[conference](https://link.springer.com/chapter/10.1007/978-3-030-93806-4_22)], 2022.

**Unifying Molecular and Textual Representations via Multi-task Language Modelling.**
*Christofidellis, Dimitrios et al.* [[abs](https://arxiv.org/abs/2301.12586)], 2023

**Is Domain Adaptation Worth Your Investment? Comparing BERT and FinBERT on Financial Tasks.**
*Peng, Bo et al.* [[proceedings](https://www.aclweb.org/anthology/2021.enlp-1.10/)], 2021

**PathologyBERT - Pre-trained Vs. A New Transformer Language Model for Pathology Domain.**
*Santos, Thiago et al.* [[proceedings](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9030007/)], 2022

**BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining.**
*Luo, Renqian et al.* [[article](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbab536/6702389)], 2022

**AraLegal-BERT: A pretrained language model for Arabic Legal text.**
*Al-Qurishi, Muhammad et al.* [[abs](https://arxiv.org/abs/2210.08284)], 2022

**ConfliBERT: A Pre-trained Language Model for Political Conflict and Violence.**
*Hu, Yibo et al.* [[conf](https://www.aclweb.org/anthology/2022.naacl-main.238/)], 2022

**MFinBERT: Multilingual Pretrained Language Model For Financial Domain.**
*Nguyen, Duong et al.* [[doi](https://doi.org/10.1109/KSE52687.2022.00006)], 2022.

**AKI-BERT: a Pre-trained Clinical Language Model for Early Prediction of Acute Kidney Injury.**
*Mao, Chengsheng et al.* [[abs](https://arxiv.org/abs/2205.03695)], 2022

**Bioformer: An Efficient Transformer Language Model for Biomedical Text Mining.**
*Fang, Li et al.* [[arXiv](https://arxiv.org/abs/)], 2023

**TourBERT: A pretrained language model for the tourism industry.**
*Arefieva, Veronika and Roman Egger.* [[abs](https://arxiv.org/abs/2201.07449)], 2022

**LeXFiles and LegalLAMA: Facilitating English Multinational Legal Language Model Development.**
*Chalkidis, Ilias et al.* [[abs](https://arxiv.org/abs/2305.07507)], 2023

**PoliBERTweet: A Pre-trained Language Model for Analyzing Political Content on Twitter.**
*Kawintiranon, Kornraphop and Lisa Singh.* [[conference](https://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.684.pdf)], 2022.

**CiteCaseLAW: Citation Worthiness Detection in Caselaw for Legal Assistive Writing.**
*Khatri, Mann et al.* [[abs](https://arxiv.org/abs/2305.03508)], 2023

**MEDBERT.de: A Comprehensive German BERT Model for the Medical Domain.**
*Bressem, Keno Kyrill et al.* [[abs](https://arxiv.org/abs/2303.08179)], 2023

**Constructing and analyzing domain-specific language model for financial text mining.**
*Suzuki, Masahiro et al.* [[doi](https://doi.org/10.1016/j.ipm.2022.103194)], 2023

**ChestXRayBERT: A Pretrained Language Model for Chest Radiology Report Summarization.**
*Cai, Xiaoyan et al.* [[doi](https://doi.org/10.1109/TMM.2022.3157645)], 2023.



### 1.2 Further Pre-Training

**Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.**
*Gururangan, Suchin et al.* [[abs](https://arxiv.org/abs/2004.10964)], 2020

**SciBERT: A Pretrained Language Model for Scientific Text.**
*Beltagy, Iz et al.* [[conf](https://www.aclweb.org/anthology/D19-1371/)], 2019

**Gradual Further Pre-training Architecture for Economics/Finance Domain Adaptation of Language Model.**
*Sakaji, Hiroki et al.* [[doi](https://doi.org/10.1109/BigData52646.2022.00098)], 2022.



### 1.3 Mixed Pre-Training

**BloombergGPT: A Large Language Model for Finance.**
*Wu, Shijie et al.* [[abs](https://arxiv.org/abs/2303.17564)], 2023



### 1.4 Domain-Specific Fine-Tuning

**Large Language Models Encode Clinical Knowledge.**
Singhal, K. et al. [[abs](https://arxiv.org/abs/2212.13138)], 2022

**Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding.**
*Toma, Augustin et al.* [[abs](https://arxiv.org/abs/2305.12031)], 2023

**ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge.**
*Li, Yunxiang et al.* [[abs](https://arxiv.org/abs/2303.14070)], 2023

**Empower Large Language Model to Perform Better on Industrial Domain-Specific Question Answering.**
*Wang, Zezhong et al.* [[abs](https://arxiv.org/abs/2305.11541)], 2023

**ExpertPrompting: Instructing Large Language Models to be Distinguished Experts.**
*Xu, Benfeng et al.* [[abs](https://arxiv.org/abs/2305.14688)], 2023

**SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models.**
*Thangarasa, Vithursan et al.* [[abs](https://arxiv.org/abs/2303.10464)], 2023

**MedJEx: A Medical Jargon Extraction Model with Wiki’s Hyperlink Span and Contextualized Masked Language Model Score.**
*Kwon, Sunjae et al.* [[proceedings](https://www.aclweb.org/anthology/2022.emnlp-main.903/)], 2022.

**Exploring the Trade-Offs: Unified Large Language Models vs Local Fine-Tuned Models for Highly-Specific Radiology NLI Task.**
*Wu, Zihao et al.* [[abs](https://arxiv.org/abs/2304.09138)], 2023

**Flan-MoE: Scaling Instruction-Finetuned Language Models with Sparse Mixture of Experts.**
*Shen, Sheng et al.* [[abs](https://arxiv.org/abs/2305.14705)], 2023

**PMC-LLaMA: Further Finetuning LLaMA on Medical Papers.**
*Wu, Chaoyi et al.* [[abs](https://arxiv.org/abs/2304.14454)], 2023



### 1.5 Domain-Specific Pre-Training Objectives

**LinkBERT: Pretraining Language Models with Document Links.**
*Yasunaga, Michihiro et al.* [[abs](https://arxiv.org/abs/2203.15827)], 2022

**Deep Bidirectional Language-Knowledge Graph Pretraining.**
*Yasunaga, Michihiro et al.* [[abs](https://arxiv.org/abs/2210.09338)], 2022

**BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks.**
*Zhang, Kaiyuan et al.* [[abs](https://arxiv.org/abs/2305.17100)], 2023

**Exploiting Language Characteristics for Legal Domain-Specific Language Model Pretraining.**
*Nair, Inderjeet and Natwar Modani.* [[Findings](https://doi.org/10.21437/Findings.2023-10)], 2023.

**Farewell to Aimless Large-scale Pretraining: Influential Subset Selection for Language Model.**
*Wang, Xiao et al.* [[abs](https://arxiv.org/abs/2305.12816)], 2023

**OPAL: Ontology-Aware Pretrained Language Model for End-to-End Task-Oriented Dialogue.**
*Chen, Zhi et al.* [[article](https://www.aclweb.org/anthology/2022.tacl-1.6/)], 2022.

**Editing Language Model-based Knowledge Graph Embeddings.**
*Cheng, Siyuan et al.* [[abs](https://arxiv.org/abs/2301.10405)], 2023

**CaseEncoder: A Knowledge-enhanced Pre-trained Model for Legal Case Encoding.**
*Ma, Yixiao et al.* [[abs](https://arxiv.org/abs/2305.05393)], 2023

**KALA: Knowledge-Augmented Language Model Adaptation.**
*Kang, Minki et al.* [[conference](https://www.aclweb.org/anthology/2022.naacl-main.68/)], 2022

**Patton: Language Model Pretraining on Text-Rich Networks.**
*Jin Bowen et al.* [[abs](https://arxiv.org/pdf/2305.12268.pdf)], 2023


### 1.6 Domain-Specific Tool Use

**GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information.**
*Jin, Qiao et al.* [[arXiv](https://arxiv.org/abs/)], 2023

**Almanac: Knowledge-Grounded Language Models for Clinical Medicine.**
*Zakka, Cyril et al.* [[abs](https://arxiv.org/abs/2303.01229)], 2023

**CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.**
*Gou, Zhibin et al.* [[abs](https://arxiv.org/abs/2305.11738)], 2023


## 2. Using Domain-Knowledge in Large Language Models

### 2.1 Black-Box Retrieval Augmentation

**CooK: Empowering General-Purpose Language Models with Modular and Collaborative Knowledge.**
*Feng, Shangbin et al.* [[abs](https://arxiv.org/abs/2305.09955)], 2023

**REPLUG: Retrieval-Augmented Black-Box Language Models.**
*Shi, Weijia et al.* [[abs](https://arxiv.org/abs/2301.12652)], 2023

**WHEN GIANT LANGUAGE BRAINS JUST AREN’T ENOUGH! DOMAIN PIZZAZZ WITH KNOWLEDGE SPARKLE DUST.**
*Nguyen, Minh-Tien et al.* [[abs](https://arxiv.org/pdf/2305.07230.pdf)], 2023

### 2.2 Retrieval-Based Pre-Training

**Knowledge-in-Context: Towards Knowledgeable Semi-Parametric Language Models.**
*Pan, Xiaoman et al.* [[abs](https://arxiv.org/abs/2210.16433)], 2022

**Atlas: Few-shot Learning with Retrieval Augmented Language Models.**
*Izacard, Gautier et al.* [[abs](https://arxiv.org/abs/2208.03299)], 2022



### 2.3 Generalist and Domain-Specific Ensembles

**Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models.**
*Li, Margaret et al.* [[abs](https://arxiv.org/abs/2208.03306)], 2022

**Scaling Expert Language Models with Unsupervised Domain Discovery.**
*Gururangan, Suchin et al.* [[abs](https://arxiv.org/abs/2303.14177)], 2023



## 3. Miscellaneous

**Scaling Data-Constrained Language Models.**
*Muennighoff, Niklas et al.* [[abs](https://arxiv.org/abs/2305.16264)], 2023

**Leveraging Domain Knowledge for Inclusive and Bias-aware Humanitarian Response Entry Classification.**
*Tamagnone, Nicolò et al.* [[abs](https://arxiv.org/abs/2305.16756)], 2023

**ChatGraph: Interpretable Text Classification by Converting ChatGPT Knowledge to Graphs.**
*Shi, Yucheng et al.* [[abs](https://arxiv.org/abs/2305.03513)], 2023

**Language Model Crossover: Variation through Few-Shot Prompting.**
*Meyerson, Elliot et al.* [[abs](https://arxiv.org/abs/2302.12170)], 2023

**Domain Knowledge Transferring for Pre-trained Language Model via Calibrated Activation Boundary Distillation.**
*Choi, Dongha et al.* [[conf](https://aclanthology.org/2022.acl-main.438/)], 2022.

**Reprogramming Pretrained Language Models for Protein Sequence Representation Learning.**
*Vinod, Ria et al.* [[abs](https://arxiv.org/abs/2301.02120)], 2023

**Reasoning with Language Model is Planning with World Model.**
*Hao, Shibo et al.* [[abs](https://arxiv.org/abs/2305.14992)], 2023

**Few-shot Learning with Retrieval Augmented Language Models.**
*Izacard, Gautier et al.* [[abs](https://arxiv.org/abs/2208.03299)], 2022

**Unified Demonstration Retriever for In-Context Learning.**
*Li, Xiaonan et al.* [[abs](https://arxiv.org/abs/2305.04320)], 2023

**AutoScrum: Automating Project Planning Using Large Language Models.**
*Schroder, Martin.* 2023.

**Explainable Automated Debugging via Large Language Model-driven Scientific Debugging.**
*Kang, Sungmin et al.* [[abs](https://arxiv.org/abs/2304.02195)], 2023

**ModuleFormer: Learning Modular Large Language Models From Uncurated Data.**
*Shen, Yikang et al.* 2023.

**Galactic ChitChat: Using Large Language Models to Converse with Astronomy Literature.**
*Ciucă, Ioana and Yuan-sen Ting.* [[abs](https://arxiv.org/abs/2304.05406)], 2023

**Grammar Prompting for Domain-Specific Language Generation with Large Language Models.**
*Wang, Bailin et al.* [[abs](https://arxiv.org/abs/2305.19234)], 2023
