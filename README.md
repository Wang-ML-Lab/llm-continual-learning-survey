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


## Continual Fine-Tuning of LLMs (CFT)

### General Continual Fine-Tuning

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
