# num - число изображений минимум миниум 10к
# images - папка до изображений
# в один процесс 10к считается в чуть меньше минуты


CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 fid.py calc --num=16 --images=./imgs --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz