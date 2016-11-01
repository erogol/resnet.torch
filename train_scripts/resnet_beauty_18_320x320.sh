th main.lua \
-depth 18 \
-batchSize 32 \
-nGPU 1 \
-nThreads 10 \
-data /media/eightbit/data_ssd/merge_354/ \
-dataset beauty_320 \
-nClasses 10 \
-resetClassifier true \
-backend cudnn \
-nEpochs 180 \
-classWeighting true \
-finetuneLastLayer false \
-LR 0.001 \
-LR_decay_step 60 \
-gen cache_files/  \
-save my_models/trained_models_beauty/ \
-shareGradInput true \
-retrain /media/eightbit/data_hdd/NNBase/Models/Torch/ImageNetResNet/resnet-18.t7 \
-testOnly false \

# -retrain /media/eightbit/data_hdd/NNBase/Models/Torch/ImageNetResNet/resnet-18.t7 \
#-retrain /media/eightbit/data_hdd/NNBase/Models/Torch/ImageNetResNet/resnet-18.t7 \
