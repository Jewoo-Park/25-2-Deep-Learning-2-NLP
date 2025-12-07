# 25-2-Deep-Learning-2-NLP

This project reproduces **"The Impact of Reasoning Step Length on Large Language Models"** (ACL 2024)

## 📄 Paper Information

- **Title**: The Impact of Reasoning Step Length on Large Language Models
- **Conference**: ACL 2024
- **Authors**: Mingyu Jin et al.
- **Paper Link**: [ACL Anthology](https://aclanthology.org/2024.findings-acl.444/)

## 🎯 Project Overview

This repository implements experiments to validate the paper's core hypothesis:

> **Longer reasoning chains improve LLM performance on complex reasoning tasks.**

We replicate both **Zero-shot Chain-of-Thought (CoT)** and **Few-shot CoT** experiments across multiple benchmarks using open-source language models.

## 🗂️ Repository Structure

```
25-2-Deep-Learning-2-NLP/
├── Few Shot/
│   ├── Mistral_7B_FewShot.ipynb          # Mistral-7B Few-shot experiments
│   ├── Qwen2_7B_FewShot.ipynb            # Qwen2-7B Few-shot experiments
│   ├── Vicuna_13B_FewShot.ipynb          # Vicuna-13B Few-shot experiments
│   └── Prompt_Perturbation.ipynb         # Prompt variation experiments
├── Zero Shot/
│   ├── Mistral_7B_ZeroShot.ipynb         # Mistral-7B Zero-shot experiments
│   ├── Qwen2_7B_ZeroShot.ipynb           # Qwen2-7B Zero-shot experiments
│   └── Vicuna_13B_ZeroShot.ipynb         # Vicuna-13B Zero-shot experiments
├── Result_Visualization.ipynb             # Visualization functions
├── Result Pics/                           # Experimental result images
└── README.md
```

## 🔬 Experimental Setup

### Models Tested

We selected three open-source models accessible without API keys:

- **Mistral-7B** (7B parameters): `mistralai/Mistral-7B-Instruct-v0.3`
- **Qwen2-7B** (7B parameters): `Qwen/Qwen2-7B-Instruct`
- **Vicuna-13B** (13B parameters): `lmsys/vicuna-13b-v1.5`

**Configuration**:
- Quantization: 4-bit (BitsAndBytes)
- Temperature: 0.0 (greedy decoding for reproducibility)
- Max tokens: 512
- Hardware: Google Colab A100 GPU

### Datasets

We evaluate on 6 reasoning benchmarks (subset of 8 from the paper):

**1. MultiArith** (Arithmetic)
- Samples: 180
- Description: Basic arithmetic word problems

**2. GSM8K** (Math Reasoning)
- Samples: 200
- Description: Grade school math problems

**3. AQuA** (Algebraic)
- Samples: 200
- Description: Multiple-choice algebra questions

**4. SVAMP** (Math Variations)
- Samples: 200
- Description: Math problem variations

**5. Letter** (Symbolic)
- Samples: 200
- Description: String manipulation (BigBench)

**6. Coin** (Object Tracking)
- Samples: 200
- Description: State tracking (BigBench)

**Missing datasets**: SingleEq (unavailable), StrategyQA (deprecated)

**Sampling**: 30% of each dataset to reduce compute time (~3 hours vs ~8 hours full)

## 🧪 Prompt Designs

### Zero-shot CoT

Three prompt conditions tested:

```python
# Baseline: No reasoning
Q: [question]
A:

# Standard CoT: Basic step-by-step
Q: [question]
A: Let's think step by step.

# Extended CoT: Explicit instruction for more steps
Q: [question]
A: Let's think step by step, you must think more steps.
```

### Few-shot CoT

Each condition includes 1-2 demonstration examples:

**Baseline** (0 reasoning steps):
```
Q: Janet has 10 apples. She gives 3 to her friend. How many apples does Janet have now?
A: The answer is 7.

Q: [actual question]
A:
```

**Standard** (2-3 reasoning steps):
```
Q: Janet has 10 apples. She gives 3 to her friend. How many apples does Janet have now?
A: Janet starts with 10 apples. She gives away 3 apples. So 10 - 3 = 7. The answer is 7.

Q: [actual question]
A:
```

**Extended** (5-6 reasoning steps):
```
Q: Janet has 10 apples. She gives 3 to her friend. How many apples does Janet have now?
A: The question is: How many apples does Janet have now?
Janet starts with 10 apples.
She gives away 3 apples to her friend.
Let me make an equation: apples_left = 10 - 3 = 7.
Self-verify: 10 - 3 = 7. This is correct.
The answer is 7.

Q: [actual question]
A:
```

### Reasoning Strategies (Extended Prompts)

Based on the paper's 5 strategies:
1. **Read the question again**: Re-state the problem
2. **Think about the word**: Analyze key terms
3. **Repeat state**: Summarize intermediate results
4. **Self-verification**: Check the answer
5. **Make equation**: Formalize the problem

## 📊 Key Results

### Zero-shot CoT Results (Mistral-7B)

**MultiArith**
- Baseline: 25.93%
- Standard CoT: 81.48% (+55.55%p) ✓ Best
- Extended CoT: 72.22% (+46.29%p)

**GSM8K**
- Baseline: 25.00%
- Standard CoT: 36.67% (+11.67%p)
- Extended CoT: 45.00% (+20.00%p) ✓ Best

**AQuA**
- Baseline: 8.33%
- Standard CoT: 25.00% (+16.67%p) ✓ Best
- Extended CoT: 23.33% (+15.00%p)

**SVAMP**
- Baseline: 3.33%
- Standard CoT: 3.33% (0.00%p)
- Extended CoT: 0.00% (-3.33%p)

**Letter**
- Baseline: 0.00%
- Standard CoT: 0.00% (0.00%p)
- Extended CoT: 0.00% (0.00%p)

**Coin**
- Baseline: 0.00%
- Standard CoT: 0.00% (0.00%p)
- Extended CoT: 0.00% (0.00%p)

**Overall Average**
- Baseline: 10.43%
- Standard CoT: 24.31% (+13.88%p)
- Extended CoT: 23.43% (+13.00%p)

### Few-shot CoT Results (Vicuna-13B)

**MultiArith**
- Baseline: 20.37%
- Standard CoT: 62.96% (+42.59%p) ✓ Best
- Extended CoT: 50.00% (+29.63%p)

**GSM8K**
- Baseline: 21.67%
- Standard CoT: 21.67% (0.00%p)
- Extended CoT: 18.33% (-3.34%p)

**AQuA**
- Baseline: 15.00%
- Standard CoT: 23.33% (+8.33%p)
- Extended CoT: 25.00% (+10.00%p) ✓ Best

**SVAMP**
- Baseline: 0.00%
- Standard CoT: 1.67% (+1.67%p)
- Extended CoT: 3.33% (+3.33%p) ✓ Best

**Letter**
- Baseline: 0.00%
- Standard CoT: 0.00% (0.00%p)
- Extended CoT: 0.00% (0.00%p)

**Coin**
- Baseline: 0.00%
- Standard CoT: 0.00% (0.00%p)
- Extended CoT: 0.00% (0.00%p)

**Overall Average**
- Baseline: 9.51%
- Standard CoT: 18.27% (+8.76%p)
- Extended CoT: 16.11% (+6.60%p)

## 🔍 Key Findings

### ✅ Supports Paper Hypothesis

1. **CoT significantly improves reasoning**: Average improvement of +13.88%p (Zero-shot) and +8.76%p (Few-shot)
2. **Extended CoT helps on harder tasks**: GSM8K improved from 36.67% → 45.00% with extended prompting
3. **Response length increases with reasoning**: Baseline (5 words) → Standard (31 words) → Extended (49 words)

### ⚠️ Contradicts Paper

1. **Extended doesn't always outperform Standard**: MultiArith showed regression (81.48% → 72.22%)
2. **No improvement on very complex tasks**: SVAMP, Letter, Coin remained at ~0% across all conditions
3. **Model dependency**: Smaller models (7B) struggle with symbolic reasoning and object tracking

### 💡 Insights

- **Task complexity matters**: CoT is most effective on arithmetic tasks (MultiArith +55%p) but fails on complex symbolic reasoning
- **Model capacity threshold**: 7B-13B models insufficient for tasks requiring dynamic programming (Letter LCS) or multi-step state tracking (Coin)
- **Prompt sensitivity**: "Think more steps" doesn't guarantee better performance; can introduce noise on simpler tasks

## 🛠️ Implementation Details

### Answer Extraction

Custom regex patterns for each dataset type:

```python
# Math datasets (GSM8K, MultiArith, SVAMP)
r'####\s*(-?\d+\.?\d*)'           # Explicit answer marker
r'[Tt]he answer is\s*(-?\d+\.?\d*)'  # Natural language
r'=\s*(-?\d+\.?\d*)(?:\s|$)'     # Last equals sign

# AQuA (multiple choice)
r'\(([A-E])\)'                    # Extract choice

# Letter (string manipulation)
r'[Ll]ength\s+(?:is|=|:)?\s*(\d+)'  # Numeric length

# Coin (object tracking)
r'has (?:the |a )?(\w+ (?:ball|present))'  # Object identification
```

### Critical Bug Fixes

1. **SVAMP context missing**: Concatenated `Body + Question` fields
2. **Letter answer format**: Extracted first element from list
3. **Coin tracking patterns**: Matched object names instead of yes/no
4. **Answer priority**: Used last match to avoid intermediate values

## 📈 Visualization

The `Result_Visualization.ipynb` notebook provides:

- **Comparative bar charts**: Accuracy across datasets and prompt types
- **Improvement heatmaps**: Standard/Extended vs Baseline gains
- **Trend analysis**: Line plots showing prompt type effects
- **Dataset difficulty ranking**: Based on baseline performance

Example usage:

```python
from visualization import analyze_and_visualize_results

results_data = [
    {'dataset': 'gsm8k', 'prompt_type': 'baseline', 'accuracy': 25.0, 'num_samples': 60},
    {'dataset': 'gsm8k', 'prompt_type': 'standard', 'accuracy': 36.67, 'num_samples': 60},
    # ... more results
]

summary_df, pivot_df, improvement_df = analyze_and_visualize_results(
    results_data, 
    model_name="Mistral-7B-v0.3"
)
```

## 🚀 Running the Experiments

### Prerequisites

```bash
pip install torch transformers datasets bitsandbytes accelerate
pip install pandas numpy matplotlib seaborn tqdm
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Jewoo-Park/25-2-Deep-Learning-2-NLP.git
cd 25-2-Deep-Learning-2-NLP
```

2. **Run Zero-shot experiments**
```python
# Open Zero Shot/Mistral_7B_ZeroShot.ipynb
# Execute all cells
```

3. **Run Few-shot experiments**
```python
# Open Few Shot/Mistral_7B_FewShot.ipynb
# Execute all cells
```

4. **Visualize results**
```python
# Open Result_Visualization.ipynb
# Run with your experimental data
```

### Runtime Estimates

- **Zero-shot** (30% sampling): ~2-3 hours per model
- **Few-shot** (30% sampling): ~3-4 hours per model
- **Full dataset**: ~8-10 hours per model

## 📝 Differences from Original Paper

**Models**
- Paper: GPT-3.5, GPT-4
- Our Implementation: Mistral-7B, Qwen2-7B, Vicuna-13B

**Datasets**
- Paper: 8 benchmarks
- Our Implementation: 6 benchmarks (2 unavailable)

**Sampling**
- Paper: Full dataset
- Our Implementation: 30% sampling

**Few-shot Demos**
- Paper: Auto-CoT generated
- Our Implementation: Manually crafted

**Reasoning Steps**
- Paper: Up to 6 steps
- Our Implementation: 2-3 (Standard), 5-6 (Extended)

## 🔮 Future Work

1. **Test larger models**: Llama-3-70B, Mixtral-8x7B for complex tasks
2. **Implement Auto-CoT**: Automatic demonstration generation via clustering
3. **Add missing datasets**: SingleEq, StrategyQA if sources become available
4. **Analyze step count**: Correlation between reasoning steps and accuracy
5. **Prompt perturbation**: Systematic variations of CoT triggers

## 📚 Citation

Original paper:
```bibtex
@inproceedings{jin-etal-2024-impact,
    title = "The Impact of Reasoning Step Length on Large Language Models",
    author = "Jin, Mingyu and others",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-acl.444"
}
```


## 📧 Contact

jw2463@g.skku.edu

## 📜 License

This project is for educational purposes as part of Deep Learning 2 (NLP) coursework.

---

**Last Updated**: December 2025
