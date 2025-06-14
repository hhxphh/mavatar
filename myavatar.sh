source ~/.bashrc
conda activate my2d-gs-avatar

source ~/.bashrc
conda activate gs-avatar

cd /media/hhx/Lenovo/code/myAvatar

tensorboard --logdir=/media/hhx/Lenovo/code/GaussianAvatarori/output/m4c_processed

tensorboard --logdir=/media/hhx/Lenovo/code/GaussianAvatarori/output/dynvideo_male

tensorboard --logdir=/media/hhx/Lenovo/code/myAvatar/output/dress_more

### Training
python trainrev.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/_4D-DRESS_00152_Inner -m output/_4D-DRESS_00152_Inner --train_stage 1 --dimension 3
python trainrev.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/_4D-DRESS_00152_Outer_1 -m output/_4D-DRESS_00152_Outer_1 --train_stage 1 --dimension 3
python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more --train_stage 3 --dimension 3 --no_mask 1
python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --train_stage 3 --dimension 3 --no_mask 1

python trainnew.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more_take2_2d --train_stage 1 --dimension 3 --no_mask 1

python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --train_stage 2 --dimension 3 --no_mask 1 --checkpoint_epochs 100
python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --train_stage 3 --dimension 3 --no_mask 1 --train_geo 1

python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my_mul -m output/dress_my_mul --train_stage 1 --dimension 3 --no_mask 1
python trainrev.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/m4c_processed -m output/m4c_processed --train_stage 1 --dimension 3

cd scripts
python gen_pose_map_our_smpl.py

cd scripts
python gen_pose_map_cano_smpl.py
python render_pred_smpl.py

python optimize_trans.py
python eval.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/m4c_processed -m output/m4c_processed --epoch 200 --train_stage 1
python render_novel_pose.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/m4c_processed -m output/m4c_processed --epoch 200 --train_stage 1  --dimension 3
python render_novel_pose.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/_4D-DRESS_00152_Inner -m output/_4D-DRESS_00152_Inner --epoch 200 --train_stage 1  --dimension 3

python render_novel_pose.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/m4c_processed -m output/m4c_processed --epoch 200 --train_stage 1  --dimension 3
python render_novel_pose.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/_4D-DRESS_00152_Inner -m output/_4D-DRESS_00152_Inner --epoch 200 --train_stage 1  --dimension 3 
python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 200 --train_stage 1  --dimension 3 
python render_novel_pose.py -s /media/hhx/Lenovo/code/GaussianAvatarori/gs_data/dress_my_mul -m output/dress_my_mul --epoch 100 --train_stage 1  --dimension 3 

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test
python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output_moretake_woinp_posmap/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test
python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output_moretake_inp_posmap/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test
python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output_trainnew_delta/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output_moretake_smplx/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test    

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test     

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my_tmp_gs --epoch 50 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001 --use_tmp_gs 1     

python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output_mayerror/dress_my_wo_poseencoder --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001 --train_smpl 0

python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test
python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test

python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001            

python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m /media/hhx/Lenovo/code/myAvatar/output_mayerror/dress_my_wo_poseencoder --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001
python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m /media/hhx/Lenovo/code/myAvatar/output_mayerror/dress_my_wo_poseencoder --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test






