# A Survey on Continual Learning in the Era of Large Language Models
An ongoing project surveying existing studies on Continual Learning x Large Language Models.

## Relevant Survey Papers
- A Comprehensive Survey of Continual Learning: Theory, Method and Application (TPAMI 2024) [[paper](https://arxiv.org/abs/2302.00487)]
- Continual Learning for Large Language Models: A Survey [[paper](https://arxiv.org/abs/2402.01364)]
- Continual Lifelong Learning in Natural Language Processing: A Survey (COLING 2020) [[paper](https://arxiv.org/abs/2012.09823)]
- Continual Learning of Natural Language Processing Tasks: A Survey [[paper](https://arxiv.org/abs/2211.12701)]
- A Survey on Knowledge Distillation of Large Language Models [[paper](https://arxiv.org/abs/2402.13116)]


## Continual Pre-Training of LLMs (CPT)
- TimeLMs: Diachronic Language Models from Twitter (ACL 2022, Demo Track) [[paper](https://arxiv.org/abs/2202.03829)][[code](https://github.com/cardiffnlp/timelms)]
- Continual Pre-Training of Large Language Models: How to (re)warm your model? [[paper](https://arxiv.org/abs/2308.04014)]
- Continual Learning Under Language Shift [[paper](https://arxiv.org/abs/2311.01200)]
- Examining Forgetting in Continual Pre-training of Aligned Large Language Models [[paper](https://arxiv.org/abs/2401.03129)]
- Towards Continual Knowledge Learning of Language Models (ICLR 2022) [[paper](https://arxiv.org/abs/2110.03215)][[code](https://github.com/joeljang/continual-knowledge-learning)]
- Lifelong Pretraining: Continually Adapting Language Models to Emerging Corpora (NAACL 2022) [[paper](https://arxiv.org/abs/2110.08534)]
- TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models (EMNLP 2022) [[paper](https://arxiv.org/abs/2204.14211)][[code](https://github.com/joeljang/temporalwiki)]
- Continual Training of Language Models for Few-Shot Learning (EMNLP 2022) [[paper](https://arxiv.org/abs/2210.05549)][[code](https://github.com/UIC-Liu-Lab/CPT)]
- ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding (AAAI 2020) [[paper](https://arxiv.org/abs/1907.12412)][[code](https://github.com/PaddlePaddle/ERNIE)]
- Dynamic Language Models for Continuously Evolving Content (KDD 2021) [[paper](https://arxiv.org/abs/2106.06297)]
- Continual Pre-Training Mitigates Forgetting in Language and Vision [[paper](https://arxiv.org/abs/2205.09357)][[code](https://github.com/AndreaCossu/continual-pretraining-nlp-vision)]
- DEMix Layers: Disentangling Domains for Modular Language Modeling (NAACL 2022) [[paper](https://arxiv.org/abs/2108.05036)][[code](https://github.com/kernelmachine/demix)]
- Time-Aware Language Models as Temporal Knowledge Bases (TACL 2022) [[paper](https://arxiv.org/abs/2106.15110)]
- Recyclable Tuning for Continual Pre-training (ACL 2023 Findings) [[paper](https://arxiv.org/abs/2305.08702)][[code](https://github.com/thunlp/RecyclableTuning)]
- Lifelong Language Pretraining with Distribution-Specialized Experts (ICML 2023) [[paper](https://arxiv.org/abs/2305.12281)]
- ELLE: Efficient Lifelong Pre-training for Emerging Data (ACL 2022 Findings) [[paper](https://arxiv.org/abs/2203.06311)][[code](https://github.com/thunlp/ELLE)]


## Domain-Adaptive Pre-Training of LLMs (DAP)
### Legal Domain
- SaulLM-7B: A pioneering Large Language Model for Law [[paper](https://arxiv.org/abs/2403.03883)][[huggingface](https://huggingface.co/papers/2403.03883)]
- Lawyer LLaMA Technical Report [[paper](https://arxiv.org/abs/2305.15062)]

### Medical Domain
- BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine [[paper](https://arxiv.org/abs/2308.09442)][[code](https://github.com/PharMolix/OpenBioMed)]
- Continuous Training and Fine-tuning for Domain-Specific Language Models in Medical Question Answering [[paper](https://arxiv.org/abs/2311.00204)]
- PMC-LLaMA: Towards Building Open-source Language Models for Medicine [[paper](https://arxiv.org/abs/2304.14454)][[code](https://github.com/chaoyi-wu/PMC-LLaMA)]
- AF Adapter: Continual Pretraining for Building Chinese Biomedical Language Model [[paper](https://arxiv.org/abs/2211.11363)]
- Continual Domain-Tuning for Pretrained Language Models [[paper](https://arxiv.org/abs/2004.02288)]
- HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs [[paper](https://arxiv.org/abs/2311.09774)][[code](https://github.com/FreedomIntelligence/HuatuoGPT-II)]

### Financial Domain
- BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark [[paper](https://arxiv.org/abs/2302.09432)][[code](https://github.com/ssymmetry/BBT-FinCUGE-Applications)]
- CFGPT: Chinese Financial Assistant with Large Language Model [[paper](https://arxiv.org/abs/2309.10654)][[code](https://github.com/TongjiFinLab/CFGPT)]
- Efficient Continual Pre-training for Building Domain Specific Large Language Models [[paper](https://arxiv.org/abs/2311.08545)]
- WeaverBird: Empowering Financial Decision-Making with Large Language Model, Knowledge Base, and Search Engine [[paper](https://arxiv.org/abs/2308.05361)][[code](https://github.com/ant-research/fin_domain_llm)][[huggingface](https://huggingface.co/weaverbirdllm)][[demo](https://www.youtube.com/watch?v=yofgeqnlrMc)]
- XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters [[paper](https://arxiv.org/abs/2305.12002)][[huggingface](https://huggingface.co/xyz-nlp/XuanYuan2.0)]

### Scientific Domain
- AstroLLaMA: Towards Specialized Foundation Models in Astronomy [[paper](https://arxiv.org/abs/2309.06126)]
- OceanGPT: A Large Language Model for Ocean Science Tasks [[paper](https://arxiv.org/abs/2310.02031)][[code](https://github.com/zjunlp/KnowLM)]
- K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization [[paper](https://arxiv.org/abs/2306.05064)][[code](https://github.com/davendw49/k2)][[huggingface](https://huggingface.co/daven3/k2-v1)]
- MarineGPT: Unlocking Secrets of "Ocean" to the Public [[paper](https://arxiv.org/abs/2310.13596)][[code](https://github.com/hkust-vgd/MarineGPT)]
- GeoGalactica: A Scientific Large Language Model in Geoscience [[paper](https://arxiv.org/abs/2401.00434)][[code](https://github.com/geobrain-ai/geogalactica)][[huggingface](https://huggingface.co/papers/2401.00434)]
- Llemma: An Open Language Model For Mathematics [[paper](https://arxiv.org/abs/2310.10631)][[code](https://github.com/EleutherAI/math-lm)][[huggingface](https://huggingface.co/EleutherAI/llemma_34b)]
- PLLaMa: An Open-source Large Language Model for Plant Science [[paper](https://arxiv.org/abs/2401.01600)][[code](https://github.com/Xianjun-Yang/PLLaMa)][[huggingface](https://huggingface.co/papers/2401.01600)]

### Code Domain
- CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis [[paper](https://arxiv.org/abs/2203.13474)][[code](https://github.com/salesforce/CodeGen)][[huggingface](https://huggingface.co/models?search=salesforce+codegen)]
- Code Needs Comments: Enhancing Code LLMs with Comment Augmentation [[code](https://arxiv.org/abs/2402.13013)]
- StarCoder: may the source be with you! [[ppaer](https://arxiv.org/abs/2305.06161)][[code](https://github.com/bigcode-project/starcoder)]
- DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence [[paper](https://arxiv.org/abs/2401.14196)][[code](https://github.com/deepseek-ai/DeepSeek-Coder)][[huggingface](https://huggingface.co/deepseek-ai)]
- IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators [[paper](https://arxiv.org/abs/2403.03894)][[code](https://github.com/UKPLab/arxiv2024-ircoder)]
- Code Llama: Open Foundation Models for Code [[paper](https://arxiv.org/abs/2308.12950)][[code]( https://github.com/facebookresearch/codellama)]

### Other Domains
- LLaMA Pro: Progressive LLaMA with Block Expansion [[paper](https://arxiv.org/abs/2401.02415)][[code](https://github.com/TencentARC/LLaMA-Pro)][[huggingface](https://huggingface.co/TencentARC/LLaMA-Pro-8B)]
- ECONET: Effective Continual Pretraining of Language Models for Event Temporal Reasoning [[paper](https://arxiv.org/abs/2012.15283)][[code](https://github.com/PlusLabNLP/ECONET)]
- Pre-training Text-to-Text Transformers for Concept-centric Common Sense [[paper](https://arxiv.org/abs/2011.07956)][[code](https://github.com/INK-USC/CALM/)][[project](https://inklab.usc.edu/calm-project/)]
- Don't Stop Pretraining: Adapt Language Models to Domains and Tasks (ACL 2020) [[paper](https://arxiv.org/abs/2004.10964)][[code](https://github.com/allenai/dont-stop-pretraining)]
- EcomGPT-CT: Continual Pre-training of E-commerce Large Language Models with Semi-structured Data [[paper](https://arxiv.org/abs/2312.15696)]
- Adapting Large Language Models via Reading Comprehension (ICLR 2024) [[paper](https://arxiv.org/abs/2309.09530)][[code](https://github.com/microsoft/LMOps)]

## Continual Fine-Tuning of LLMs (CFT)

### General Continual Fine-Tuning
- Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning (NeurIPS 2021) [[paper](https://arxiv.org/abs/2112.02706)][[code](https://github.com/ZixuanKe/PyContinual)]
- Can BERT Refrain from Forgetting on Sequential Tasks? A Probing Study (ICLR 2023) [[paper](https://arxiv.org/abs/2303.01081)][[code](https://github.com/kobayashikanna01/plms_are_lifelong_learners)]
- CIRCLE: Continual Repair across Programming Languages (ISSTA 2022) [[paper](https://arxiv.org/abs/2205.10956)]
- ConPET: Continual Parameter-Efficient Tuning for Large Language Models [[paper](https://arxiv.org/abs/2309.14763)][[code](https://github.com/Raincleared-Song/ConPET)]
- Enhancing Continual Learning with Global Prototypes: Counteracting Negative Representation Drift [[paper](https://arxiv.org/abs/2205.12186)]
- Investigating Forgetting in Pre-Trained Representations Through Continual Learning [[paper](https://arxiv.org/abs/2305.05968)]
- Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models [[paper](https://arxiv.org/abs/2312.07887)][[code](https://github.com/zzz47zzz/pretrained-lm-for-incremental-learning)]
- LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5 (ICLR 2022) [[paper](https://arxiv.org/abs/2110.07298)][[code](https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning)]
- On the Usage of Continual Learning for Out-of-Distribution Generalization in Pre-trained Language Models of Code [[paper](https://arxiv.org/abs/2305.04106)]
- Overcoming Catastrophic Forgetting in Massively Multilingual Continual Learning (ACL 2023 Findings) [[paper](https://arxiv.org/abs/2305.16252)]
- Parameterizing Context: Unleashing the Power of Parameter-Efficient Fine-Tuning and In-Context Tuning for Continual Table Semantic Parsing (NeurIPS 2023) [[paper](https://arxiv.org/abs/2310.04801)][[code](https://github.com/KSESEU/C3)]

### Continual Instruction Tuning (CIT)
- Fine-tuned Language Models are Continual Learners [[paper](https://arxiv.org/pdf/2205.12393.pdf)]
- TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models [[paper](https://arxiv.org/pdf/2310.06762.pdf)][[code](https://github.com/BeyonderXX/TRACE)]
- Large-scale Lifelong Learning of In-context Instructions and How to Tackle It [[paper](https://aclanthology.org/2023.acl-long.703.pdf)]
- CITB: A Benchmark for Continual Instruction Tuning [[paper](https://arxiv.org/pdf/2310.14510.pdf)]
- Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal [[paper](https://arxiv.org/pdf/2403.01244.pdf)]
- Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning [[paper](https://arxiv.org/pdf/2403.10056.pdf)]
- ConTinTin: Continual Learning from Task Instructions [[paper](https://arxiv.org/pdf/2203.08512.pdf)]
- Orthogonal Subspace Learning for Language Model Continual Learning [[paper](https://arxiv.org/pdf/2310.14152.pdf)]
- SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models [[paper](https://arxiv.org/pdf/2401.08295.pdf)]
- InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with Instructions [[paper](https://arxiv.org/pdf/2403.11435.pdf)]

### Continual Model Refinement (CMR)
- Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors [[paper](https://arxiv.org/pdf/2211.11031.pdf)][[code](https://github.com/thartvigsen/grace)]
- On Continual Model Refinement in Out-of-Distribution Data Streams [[paper](https://arxiv.org/pdf/2205.02014.pdf)][[code](https://github.com/facebookresearch/cmr)][[project](https://cmr-nlp.github.io/)]
- Melo: Enhancing model editing with neuron-indexed dynamic lora [[paper](https://arxiv.org/pdf/2312.11795.pdf)][[code](https://github.com/ECNU-ICALK/MELO)]
- Larimar: Large language models with episodic memory control [[paper](https://arxiv.org/pdf/2403.11901.pdf)]
- Wilke: Wise-layer knowledge editor for lifelong knowledge editing [[paper](https://arxiv.org/pdf/2402.10987.pdf)]


### Continual Model Alignment (CMA)
- Alpaca: A Strong, Replicable Instruction-Following Model [[project](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[code](https://github.com/tatsu-lab/stanford_alpaca)]
- Self-training Improves Pre-training for Few-shot Learning in Task-oriented Dialog Systems [[paper](https://arxiv.org/pdf/2108.12589.pdf)] [[code](https://github.com/MiFei/ST-ToD)]
- Training language models to follow instructions with human feedback (NeurIPS 2022) [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)]
- Direct preference optimization: Your language model is secretly a reward model (NeurIPS 2023) [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)]
- Copf: Continual learning human preference through optimal policy fitting [[paper](https://arxiv.org/pdf/2310.15694)]
- CPPO: Continual Learning for Reinforcement Learning with Human Feedback (ICLR 2024) [[paper](https://openreview.net/pdf?id=86zAUE80pP)]
- A Moral Imperative: The Need for Continual Superalignment of Large Language Models [[paper](https://arxiv.org/pdf/2403.14683)]
- Mitigating the Alignment Tax of RLHF [[paper](https://arxiv.org/abs/2309.06256)]

### Continual Multimodal LLMs (CMLLMs)
- Investigating the Catastrophic Forgetting in Multimodal Large Language Models (PMLR 2024) [[paper](https://arxiv.org/abs/2309.10313)]
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models [[paper](https://arxiv.org/abs/2304.10592)] [[code](https://github.com/Vision-CAIR/MiniGPT-4)]
- Visual Instruction Tuning (NeurIPS 2023, Oral) [[paper](https://arxiv.org/abs/2304.08485)] [[code](https://github.com/haotian-liu/LLaVA)]
- Continual Instruction Tuning for Large Multimodal Models [[paper](https://arxiv.org/abs/2311.16206)]
- CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model [[paper](https://arxiv.org/abs/2403.08350)] [[code](https://github.com/zackschen/coin)]
- Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models [[paper](https://arxiv.org/abs/2402.12048)]
- Reconstruct before Query: Continual Missing Modality Learning with Decomposed Prompt Collaboration [[paper](https://arxiv.org/abs/2403.11373)] [[code](https://github.com/Tree-Shu-Zhao/RebQ.pytorch)]
