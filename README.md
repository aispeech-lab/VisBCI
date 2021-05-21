# VisBCI
Pytorch codes of our paper [*"Improving Cross-State and Cross-Subject Visual ERP-based BCI with Temporal Modeling and Adversarial Training"*](https://github.com/aispeech-lab/VisBCI)

## A Simple Introduciton 
Brain-computer interface (BCI) is a useful device for people without relying on peripheral nerves and muscles. However, the performance of the event-related potential (ERP)-based BCI declines when applying it to the real environments, especially in cross-state and cross-subject conditions. 

For addressing this problem, we propose a general method with **hierarchical temporal modeling** and **adversarial training** to enhance the performance of the visual ERP-based BCI under different levels of auditory workload states. The artitecture of our model is shown below.

![](https://github.com/aispeech-lab/VisBCI/blob/main/assets/framework.pdf)

## The Results                                                                
We conduct our experiments on one published visual ERP-based BCI task with 10 subjects and 3 different auditory workload states. The results demonstrate that our method can work effectively under different situations and achieve satisfactory performance compared with the only subject-specific baseline. In the single-subject task and mixed-subject task, the averaged result of all the repetitions and states overpass the baseline by 6.54% and 6.66%, respectively, and only slight 0.15% is declined in the cross-subject task.

If you have the interest in our work, please further see the whole paper with this link.
