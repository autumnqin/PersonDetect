CUDA_VISIBLE_DEVICE=2 python ./faster_rcnn/train_net.py --gpu 2 --weights /home1/data/pva9.1_preAct_train_iter_1900000.npy --imdb voc_2012_trainval --iters 2000000 --cfg ./experiments/cfgs/faster_rcnn_end2end_pva.yml --network PVAnet_train --set EXP_DIR ./out_pva 2>&1 | tee pva_log.txt
