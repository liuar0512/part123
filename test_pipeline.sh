CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt ckpt/syncdreamer-pretrain.ckpt --input testdata/bear.png --output output/mvimgs/bear --elevation 30  --crop_size 200
CUDA_VISIBLE_DEVICES=0 python sam_amg.py --checkpoint ckpt/sam_vit_h_4b8939.pth --model-type default
CUDA_VISIBLE_DEVICES=0 python train_part123.py -i output/mvimgs/bear/mvout.png -n bear -b configs/neus_cw.yaml -l output/renderer
CUDA_VISIBLE_DEVICES=0 python test_part123.py -i output/mvimgs/bear/mvout.png -n bear -b configs/neus_cw.yaml -l output/renderer -r

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt ckpt/syncdreamer-pretrain.ckpt --input testdata/chicken.png --output output/mvimgs/chicken --elevation 30  --crop_size 200
CUDA_VISIBLE_DEVICES=0 python sam_amg.py --checkpoint ckpt/sam_vit_h_4b8939.pth --model-type default
CUDA_VISIBLE_DEVICES=0 python train_part123.py -i output/mvimgs/chicken/mvout.png -n chicken -b configs/neus_cw.yaml -l output/renderer
CUDA_VISIBLE_DEVICES=0 python test_part123.py -i output/mvimgs/chicken/mvout.png -n chicken -b configs/neus_cw.yaml -l output/renderer -r
