# VisBCI
Pytorch codes of our paper [*"Improving Cross-State and Cross-Subject Visual ERP-based BCI with Temporal Modeling and Adversarial Training"*](https://github.com/aispeech-lab/VisBCI)

## A Simple Introduciton 
Brain-computer interface (BCI) is a useful device for people without relying on peripheral nerves and muscles. However, the performance of the event-related potential (ERP)-based BCI declines when applying it to the real environments, especially in cross-state and cross-subject conditions. 

For addressing this problem, we propose a general method with **hierarchical temporal modeling** and **adversarial training** to enhance the performance of the visual ERP-based BCI under different auditory workload states. The artitecture of our model is shown below.

<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/framework.jpg"></div>
 
The rationality of our method here is that the ERP-based BCI is based on electroencephalography (EEG) signals recorded from the surface of the scalp, which are continuously changed with time and somewhat stochastic. Therefore, the hierarchical recurrent network is proposed to encode each ERP signal and model them in each repetition with a temporal manner to predict which visual event elicited an ERP, while adversarial training is useful to capture more properties of stochasticity acting the same as increasing training data.

## Result                                                                
We conduct our experiments on one published visual ERP-based BCI task with 10 subjects and 3 different auditory workload states. The results demonstrate that our method can work effectively under different situations and achieve satisfactory performance compared with the only subject-specific baseline. In the single-subject task and mixed-subject task, the averaged result of all the repetitions and states overpass the baseline by 6.54% and 6.66%, respectively, and only slight 0.15% is declined in the cross-subject task.  
<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/acc-all.jpg"></div>

## Cite us
If you have the interest in our work, or use this code or part of it, please cite us!   
For more detailed descirption, you can further explore the whole paper with [this link](https://github.com/aispeech-lab/VisBCI).

