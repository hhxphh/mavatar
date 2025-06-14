source ~/.bashrc
conda activate gs-avatar
conda activate my2d-gs-avatar

cd /media/hhx/Lenovo/code/myAvatar

tensorboard --logdir=/media/hhx/Lenovo/code/myAvatar/output/dress_simple_2d_new_gs_predall 
tensorboard --logdir=/media/hhx/Lenovo/code/myAvatar/output/dress_simple_2_predall_nlpls

python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --train_stage 3 --dimension 3 --no_mask 1

python train2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_moretake2f_female_smplx --train_stage 3 --dimension 3 --no_mask 1 --train_geo 1

python train2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more_2d_take2f_female_smplx --train_stage 3 --dimension 3 --no_mask 1 --train_geo 1
python train2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my_2d_take2f_female_smplx --train_stage 3 --dimension 3 --no_mask 1 --train_geo 1
python train2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple -m output/dress_simple_2_nlpls --train_stage 3 --dimension 2 --no_mask 1 --train_geo 1
python train2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple -m output/dress_simple_2_predall --train_stage 3 --dimension 2 --no_mask 1 --train_geo 1
python train2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more_2d_predall --train_stage 3 --dimension 2 --no_mask 1 --train_geo 1

python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple_2_nlpls --epoch 200 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0001 
python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple_2_nlpls --epoch 200 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test

python render_novel_pose_new_gs_2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple_2_predall_nlpls  --epoch 115 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0004 

python train_new_gs_2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple -m output/dress_simple_2d_new_gs_predall --train_stage 3 --dimension 2 --no_mask 1 --train_geo 1

python train_new_gs_2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple -m output/dress_simple_2_predall_nlpls --train_stage 3 --dimension 2 --no_mask 1 --train_geo 1
python render_novel_pose_new_gs_2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple_2d_new_gs_predall --epoch 200 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0004 

python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple --epoch 100 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0004 
python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple_2_predall --epoch 55 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0004 
python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_simple_2_predall --epoch 55 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple/train/take2/0002 

python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more_2d_predall --epoch 40 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0004      

python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_more_2d_predall --epoch 40 --train_stage 3  --dimension 2 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test/0001
 
python trainnew.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --train_stage 1 --dimension 3 --no_mask 1

python trainrev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --train_stage 2 --dimension 3 --no_mask 1 --checkpoint_epochs 100

cd scripts
python gen_pose_map_our_smpl.py

cd scripts
python gen_pose_map_cano_smpl.py
python render_pred_smpl.py

python optimize_trans.py
python eval.py -s /media/hhx/Lenovo/code/GaussianAvatar/gs_data/m4c_processed -m output/m4c_processed --epoch 200 --train_stage 1

python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_moretake2f_female_smplx --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001
python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_moretake2f_female_smplx --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test
python render_novel_pose2f.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output/dress_moretake2f_female_smplx --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2/0001  

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 200 --train_stage 1  --dimension 3 
python render_novel_pose.py -s /media/hhx/Lenovo/code/GaussianAvatarori/gs_data/dress_my_mul -m output/dress_my_mul --epoch 100 --train_stage 1  --dimension 3 

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/test

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my_tmp_gs --epoch 50 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001 --use_tmp_gs 1     

python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output/dress_my --epoch 100 --train_stage 3  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001
python render_novel_poserev.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more -m output_mayerror/dress_my_wo_poseencoder --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test

python render_novel_pose.py -s /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my -m output_mayerror/dress_my_wo_poseencoder --epoch 100 --train_stage 1  --dimension 3 --test_folder /media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my/train/0001 --train_smpl 0





