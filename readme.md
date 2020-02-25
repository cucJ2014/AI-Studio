## 算法思路

### A. 梯度攻击算法：
结合以下一些迁移性比较好的梯度攻击算法：  
MI-FGSM  
Input Diversity  
Ensemble Attack  
以无目标攻击为主要任务，攻击的损失函数为CE Loss 。动量、输入变换和融合 logits 对攻击效果的提升非常明显  

### B. 攻击的模型  
ResNeXt50_32x4d  
MobileNetV2_x2_0  
InceptionV4  
ResNet50  
EfficientNetB0  
ResNeXt50_64x4d  
MobileNetV1_x1_0  


### C. 参考文献
[1] Xie, Cihang, et al. "Feature denoising for improving adversarial robustness." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.  
[2] Dong, Yinpeng, et al. "Boosting adversarial attacks with momentum." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.  
[3] Xie, Cihang, et al. "Improving transferability of adversarial examples with input diversity." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.  
[4] Xie, Cihang, et al. "Mitigating adversarial effects through randomization." arXiv preprint arXiv:1711.01991 (2017).  



