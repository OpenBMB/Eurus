<div align="center">

<!-- <img src="figures/logo.png" width="400px"> -->

**Eurus: A suit of open-source LLMs optimized for reasoning**

<p align="center">
 <a href="#introduction"> Introduction</a> •
  <a href="#evaluation">Evaluation</a> •
</p>


</div>


# Links

- 📜 [Paper]()
- 🤗 [Eurus Collection]()
- 🤗 [UltraInteract]()

# Introduction

## Eurus

We release a suite of LLMs and a reward model. **Eurus-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks**, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. We also train a reward model that demonstrates especially strong preference modeling performance on reasoning tasks. 

- *Eurus-7B-SFT* and *Eurus-70B-SFT*: Fine-tuned from Mistral-7B and CodeLLaMA-70B on all correct actions in UltraInteract, mixing a small proportion of UltraChat, ShareGPT, and OpenOrca examples.
- *Eurus-7B-KTO* and *Eurus-70B-NCA*: Preference fine-tuned on UltraInteract and UltraFeedback on top of SFT models.
- *Eurus-RM-7B*: Trained on a mixture of UltraInteract, UltraFeedback, and UltraSafet.

<img src="figures/lc_tqa.png" width="800px">


## UltraInteract

The strong performance of Eurus can be primarily attributed to UltraInteract, a large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. 

Conceptually, UltraInteract collects a preference tree for each instruction, with the instruction being the root and each action a node (Figure 2). A trajectory is a root-to-leaf path consisting of a sequence of actions. In each preference tree, all nodes of correct actions and all trajectories ending with correct actions can be used for SFT. Paired correct and incorrect nodes or trajectories can be used for preference learning.

## Structure
<img src="figures/tree.png" width="500px">

## Illustrative Example
<img src="figures/ui_example.png" width="800px">

## Stats
<img src="figures/stats.png" width="800px">


# Evaluation

## Eurus-7B and Eurus-70B
- EURUS, both the 7B and 70B variants, achieve the best overall performance among open-source models of similar sizes. EURUS even outperforms specialized models in corresponding domains in many cases. Notably, EURUS-7B outperforms baselines that are 5× larger, and EURUS-70B achieves better performance than GPT-3.5 Turbo. 
- Preference learning with UltraInteract can further improve performance, especially in math and the multi-turn ability.

<img src="figures/main_exp.png" width="800px">

## Eurus-RM-7B
- EURUS-RM-7B stands out as the best 7B RM overall and achieves similar or better performance than much larger baselines. Particularly, it outperforms GPT-4 in certain tasks.
- Our training objective is beneficial in improving RM performance on hard problems and reasoning.
- ULTRAINTERACT is compatible with other datasets like UltraFeedback and UltraSafety, and mixing these datasets can balance different RM abilities.
- EURUS-RM-7B improves LLMs’ reasoning performance by a large margin through reranking.

<img src="figures/rm_exp.png" width="800px">



# Dataset Format
```jsonc
{

}
```








## Citation
```bib
@misc{yuan2024advancing,
      title={Advancing LLM Reasoning Generalists with Preference Trees}, 
      author={Lifan Yuan and Ganqu Cui and Hanbin Wang and Ning Ding and Xingyao Wang and Jia Deng and Boji Shan and Huimin Chen and Ruobing Xie and Yankai Lin and Zhenghao Liu and Bowen Zhou and Hao Peng and Zhiyuan Liu and Maosong Sun},
      year={2024},
      primaryClass={cs.CL}
}
```
