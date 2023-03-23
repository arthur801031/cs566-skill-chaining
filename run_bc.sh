CUDA_VISIBLE_DEVICES=0 python bc.py \
--bc_data=bc_data/chair_ingolf_0650.gail.p1.123_step_00017203200_2000_trajs.pkl \
--demo_path=demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 \
--run_prefix=p1 \
--algo=gail \
--preassembled=0 \
--run_name_postfix=policy2 \
--wandb=True