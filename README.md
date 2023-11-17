# inception
Я решил повторить пройденные мною ранее архитектуры свёрточных нейросетей и сделать небольшой проект, в котором мною была поставлена: цель реализовать GoogLeNet, протестировать её на каком-нибудь небанальном датасете(потому что использовать MNIST или cats vs dogs уже становится мовитоном), а также реализовать свой проект не как Jupyter Notebook, а как многофайловый проект.

Мою реализацию сложно назвать именно GoogLeNet, т.к. я использовал батч-нормализацию, которая появилась только в Inception-V2, а также у меня нет дополнительных выходов сети, которые уменьшали вероятность затухания градиентов. 

Модель обучал на видеокарте NVIDIA RTX 3070ti на своём домашнем ПК. Размер батча взял 128, обучал 50 эпох, но лучшая точность была у 40 эпох (0.8 на трейне и 0.68 на тесте). 

Проект будет дополняться, пока что я не посчитал top-5 accuracy, можно получить лучше точность на тесте, добавить дополнительные выходы. Также в планах добавить skip connections, как в ResNet. Тогда архитектура моей сети будет очень похожа на Inception V3

