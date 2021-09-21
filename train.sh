CUDA_VISIBLE_DEVICES=0,3,5,6 python -m torch.distributed.launch --nproc_per_node=4\
	--master_port=1234 main_train_psnr.py\
	--opt options/swinir/train_swinir_sr_classical.json\
	--dist True
