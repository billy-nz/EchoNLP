## Overview
This study applied a deep learning model to classify levels of left ventricular ejection fraction using free-text data recorded in 965 echocardiography (ECHO) reports from Auckland City and Middlemore Hospitals. Currently, clinicians are required to manually review a patient’s ECHO report to determine their level of left ventricular ejection fraction (LVEF). While several efforts have been made to automatically extract information from ECHO reports, none have adopted a deep learning approach. This study aimed to establish some baseline performance measures using a 1-dimensional convolutional neural network (CNN) and determine strategies for improving classification precision. The results were promising but were unable to exceed precision or F-scores of 80% for mild, moderate, or severe LVEF; irrespective of how the training data was sampled, how deeply connected the neural network layers were, or well the model was tuned. This study strongly supports the need to develop a chunking algorithm that could help captures conceptual, contextual, and measurement features from ECHO reports.

## Initial Results
Despite best efforts, the precision for detecting each of the three abnormal classes of LV function (mild, moderate, and severe) was disappointing. Clearly, the advantage of ‘resampling with replacement' could not induce a F1-score beyond 80% for any of the three classes. Furthermore, adding additional features to the embedding matrix such as a TF-IDF weighted matrix (either as uni-gram or n-gram) or an LDA topic distribution matrix, have little effect on improving precision or F-score beyond those reported in the results. This lack of performance improvement shows that adding more features does not necessary equate to better precision. On the contrary, adding too many features could induce more noise into model training.
 
See [documentation](https://github.com/billy-nz/EchoNLP/tree/master/doc) for complete study report or [log files](https://github.com/billy-nz/EchoNLP/tree/master/log) for raw FPR results (F1-Precision-Recall).

![picture](/images/Normal_9689.png)
Figure 1 - Accuracy and loss curves (training versus testing) for normal LV function
<br><br>
![picture](/images/Mild_9171.png)
Figure 3 - Accuracy and loss curves (training versus testing) for mild LV function
<br><br>
![picture](/images/Moderate_9223.png)
Figure 3 - Accuracy and loss curves (training versus testing) for moderate LV function
<br><br>
![picture](/images/Severe_9793.png)
Figure 4 - Accuracy and loss curves (training versus testing) for Severe LV function
<br><br>
