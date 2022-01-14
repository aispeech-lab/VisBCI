# VisBCI
Pytorch codes of our paper [*"Improving Cross-State and Cross-Subject Visual ERP-based BCI with Temporal Modeling and Adversarial Training"*](https://github.com/aispeech-lab/VisBCI), which is undergoing peer-review. 

## A Simple Introduciton 
Brain-computer interface (BCI) is a useful device for people without relying on peripheral nerves and muscles. However, the performance of the event-related potential (ERP)-based BCI declines when applying it to the real environments, especially in cross-state and cross-subject conditions. 

For addressing this problem, we propose a general method with **hierarchical temporal modeling** and **adversarial training** to enhance the performance of the visual ERP-based BCI under different auditory workload states. The artitecture of our model is shown below.

<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/framework.jpg"></div>
 
The rationality of our method here is that the ERP-based BCI is based on electroencephalography (EEG) signals recorded from the surface of the scalp, continuously changing with time and somewhat stochastic. Therefore, the hierarchical recurrent network is proposed to encode each ERP signal and model them in each repetition with a temporal manner to predict which visual event elicited an ERP, while adversarial training is useful to capture more properties of stochasticity acting the same as increasing training data.

## Result                                                                
We conduct our experiments on one published visual ERP-based BCI task with 10 subjects and 3 different auditory workload states. The results demonstrate that our method can work effectively under different situations and achieve satisfactory performance compared with the only subject-specific baseline. In the specific-subject task and mixed-subject task, the averaged result of all the repetitions and states overpass the baseline by 6.54% and 6.66%, respectively, and only slight 0.15% is declined in the cross-subject task. Direct comparison between the performance of our model and the original baseline in specific-subject, mixed-subject and cross-subject tasks is as follows. 
<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/acc-all.png"></div>

The more direct comparison between the mixed-subject task and specific-subject task under all repetitions is reported here, as the supplementary material of our paper. It is easy to see that the results obtained in the mixed-subject task are even better than the results of the specific-subject task in many cases.
<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/mixed_compare_specific.png"></div>

There are two subjects performing not well in almost all tasks, here we report the performance of our model after removing these two "bad" subjects. The updated comparison of the results of 13 subjects with original grand-averaged results of 15 subjects, and ERP waveforms in the six most relevant channels (‘PO3’, ‘POZ’, ‘PO4’, ‘O1’, ‘OZ’, ‘O2’) of these subjects are also shown below.
<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/specific-2.png"></div>
<div align=center><img src="https://github.com/aispeech-lab/VisBCI/blob/main/assets/cross-2.png"></div>

## Cite us
If you have the interest in our work, or use this code or part of it, please cite us!  
Consider citing:
```bash
@article{ni2021bci,
  title={Improving Cross-State and Cross-Subject  
Visual ERP-based BCI with Temporal Modeling and Adversarial Training},
  author={Ni, Ziyi and Xu, Jiaming and Wu, Yuwei and Li, Mengfan and Xu, Guizhi and Xu, Bo}
}
```
For more detailed descirption, you can further explore the whole paper with [this link](https://github.com/aispeech-lab/VisBCI).  

## Contact us
Any other questions, feel free to contact us at niziyi2021@ia.ac.cn or jiaming.xu@ia.ac.cn 


