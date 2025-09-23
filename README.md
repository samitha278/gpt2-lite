# gpt2-lite



A reproduction of GPT-2 from scratch. Includes the network architecture, optimized training pipeline and hyperparameters inspired by the GPT-2 and GPT-3 papers.


---

## GPT-2 Architecture
<p align="center">
  <img src="images/gpt2.png" alt="GPT-2 architecture" width="400"/><br>
  <sub>Source: <a href="https://arxiv.org/pdf/2305.07716">arXiv:2305.07716</a></sub>
</p>

---

## GPT-2 Forward Pass
<p align="center">
  <img src="images/gpt2_ff.png" alt="GPT-2 forward pass" width="400"/><br>
  <sub>Source: <a href="https://arxiv.org/pdf/2305.07716">arXiv:2305.07716</a></sub>
</p>



---
## Configuration
```
n_layer    = 12
n_head     = 12 
n_embd     = 768 
vocab_size = 50257 
block_size = 1024
```

---

## Inference Example (Hugging Face GPT-2 weights in custom model)

Prompt:
```
Hello World ! I'm LLM
```

Generated Output (sample): 5 sequences 32 max tokens
```
Hello World ! I'm LLM In ( is an no. The more 0 and you, at by the and for not the: of not 
Hello World ! I'm LLM, in... (.. The to a. : " , have the 1 as 
Hello World ! I'm LLM. 1 was ( (1. A " ( for The is on by from the of by that.. 
Hello World ! I'm LLM's " of in this and and- of, they have the the only a and that on as that all to on that 
Hello World ! I'm LLM was being the of the and on the you just, one (: the most I. It you have or a to
```


---





## Progress


### Step 1 

**Training Configuration**
- Max Iterations: 1000  
- Learning Rate: 3e-4  
- Batch Size: 32  
- Context Length (T): 8 


<div style="display: flex; align-items: flex-start; gap: 20px;">

<div>


**Training Results:**
```
 0/1000   11.001113891601562
100/1000   4.941615581512451
200/1000   4.233631134033203
300/1000   5.208221435546875
400/1000   4.291534423828125
500/1000   4.187107086181641
600/1000   6.156216144561768
700/1000   5.745358467102051
800/1000   4.903677463531494
900/1000   6.526307106018066
```
</div>

<div>

**Loss Curve**  
<img src="images/s1.png" alt="Loss curve - Step 1" width="350"/>

</div>

</div>





---

## References

- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
- [Language Models are Few-Shot Learners (GPT-3 paper)](https://arxiv.org/abs/2005.14165)
