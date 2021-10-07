#export CUDA_VISIBLE_DEVICES=7
#python main.py --exp_name=train_hyperbolic_gpu --num_points=1024 --use_sgd=True --batch_size 32 --epochs 1000 --lr 0.0001 --hyperbolic --fps_gpu=True

#export CUDA_VISIBLE_DEVICES=6
#python main.py --exp_name=train_hyperbolic_cpu --num_points=1024 --use_sgd=True --batch_size 32 --epochs 1000 --lr 0.0001 --hyperbolic


#export CUDA_VISIBLE_DEVICES=5
#python main.py --exp_name=train_hyperbolic_gpu --num_points=1024 --use_sgd=True --batch_size 64 --epochs 250 --lr 0.001 --hyperbolic --fps_gpu=True

#export CUDA_VISIBLE_DEVICES=4
#python main.py --exp_name=train_hyperbolic_cpu --num_points=1024 --use_sgd=True --batch_size 64 --epochs 250 --lr 0.001 --hyperbolic
