
CUDA=2 && SEED=0 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=2 && SEED=1 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=2 && SEED=2 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=2 && SEED=3 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=2 && SEED=4 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done

CUDA=3 && SEED=5 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=3 && SEED=6 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=3 && SEED=7 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=3 && SEED=8 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done
CUDA=3 && SEED=9 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python main.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 | tee logs/20211003/supervised/mislabel"$RATIO"_seed"$SEED".log; done


# CUDA=4 && SEED=0 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=4 && SEED=1 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=4 && SEED=2 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=4 && SEED=3 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=4 && SEED=4 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done

# CUDA=5 && SEED=5 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=5 && SEED=6 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=5 && SEED=7 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=5 && SEED=8 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done
# CUDA=5 && SEED=9 && for RATIO in $(seq 0.1 0.1 1.0); do CUDA_VISIBLE_DEVICES=$CUDA python self_supervised.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-epochs 10 | tee logs/20211003/self/mislabel"$RATIO"_seed"$SEED".log; done


# CUDA=2 && SEED=0 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=3 && SEED=1 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=4 && SEED=2 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=5 && SEED=3 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=6 && SEED=4 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done

# CUDA=7 && SEED=5 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=7 && SEED=6 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=3 && SEED=7 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=4 && SEED=8 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done
# CUDA=5 && SEED=9 && for RATIO in $(seq 0.1 0.1 0.9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log &\;; done



# CUDA=6 && SEED=0 && RATIO=0.2 && CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log
# CUDA=7 && SEED=4 && RATIO=0.5 && CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log
# CUDA=6 && SEED=3 && RATIO=0.6 && CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log
# CUDA=7 && SEED=2 && RATIO=0.7 && CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log
# CUDA=6 && SEED=0 && RATIO=0.8 && CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log
# CUDA=7 && SEED=5 && RATIO=0.8 && CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5 | tee logs/20211003/scl/mislabel"$RATIO"_seed"$SEED".log


RATIO=0.6 && for SEED in $(seq 0 9); do CUDA=$(($SEED%6+2)); CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 10 --pretrain-discriminator-epochs 0 --weight-factor 0.5 2>&1 | tee logs/20211004/scl/mislabel"$RATIO"_modelpretrain10_seed"$SEED".log &\;; done


RATIO=0.6 && for SEED in $(seq 9 -1 0); do CUDA=$(($SEED%6+2)); CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 10 --pretrain-discriminator-epochs 10 --weight-factor 0.5 2>&1 | tee logs/20211004/scl/mislabel"$RATIO"_modelpretrain10_discriminatorpretrin10_seed"$SEED".log &\;; done


CUDA_VISIBLE_DEVICES=7 python scl.py --seed 0 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 0.5



RATIO=0.6 && for ALPHA in $(seq 0.1 0.1 0.8); do for SEED in $(seq 0 9); do CUDA=$(($SEED%5+1)); CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor $ALPHA 2>&1 | tee logs/20211004/scl_alpha/mislabel"$RATIO"_alpha"$ALPHA"_seed"$SEED".log &\;; done; done

ALPHA=0.9 && CUDA=0 && RATIO=0.6 && for SEED in $(seq 0 9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor $ALPHA 2>&1 | tee logs/20211004/scl_alpha/mislabel"$RATIO"_alpha"$ALPHA"_seed"$SEED".log &\;; done

ALPHA=1.0 && CUDA=7 && RATIO=0.6 && for SEED in $(seq 0 9); do CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor $ALPHA 2>&1 | tee logs/20211004/scl_alpha/mislabel"$RATIO"_alpha"$ALPHA"_seed"$SEED".log &\;; done



CUDA_VISIBLE_DEVICES=1 python scl.py --seed 0 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=2 python scl.py --seed 1 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=3 python scl.py --seed 2 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=4 python scl.py --seed 3 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=5 python scl.py --seed 4 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=1 python scl.py --seed 5 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=2 python scl.py --seed 6 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=3 python scl.py --seed 7 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=4 python scl.py --seed 8 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0
CUDA_VISIBLE_DEVICES=5 python scl.py --seed 9 --data-ratio 0.6 --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor 1.0



RATIO=0.6 && for ALPHA in $(seq 0.1 0.1 0.9); do for SEED in $(seq 0 9); do CUDA=$(($SEED%5+1)); CUDA_VISIBLE_DEVICES=$CUDA python scl.py --seed $SEED --data-ratio $RATIO --mislabel-sample --noise-sample-ratio 0.1 --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 --weight-factor $ALPHA 2>&1 | tee logs/20211004/scl_alpha/mislabel"$RATIO"_alpha"$ALPHA"_seed"$SEED".log; done; done


RATIO=$1
CUDA=$2
MAX_PROC_CNT=$((13+$3))
for ALPHA in $(seq 0.1 0.1 0.9); do
    for SEED in $(seq 0 9); do
        while [[ $(gpustat | grep "\[1\]" | awk '{print NF}') >= $MAX_PROC_CNT ]]; do
            sleep 10
        done
        CUDA_VISIBLE_DEVICES=$CUDA python scl.py \
            --seed $SEED --data-ratio $RATIO \
            --mislabel-sample --noise-sample-ratio 0.1 \
            --pretrain-model-epochs 0 --pretrain-discriminator-epochs 0 \
            --weight-factor $ALPHA 2>&1 \
            | tee logs/20211004/scl_alpha/mislabel"$RATIO"_alpha"$ALPHA"_seed"$SEED".log &
    done
done


# ./run.sh 0.1 0 12
# ./run.sh 0.2 1 16
# ./run.sh 0.3 2 16
# ./run.sh 0.4 3 16
# ./run.sh 0.5 4 16
# ./run.sh 0.7 5 16
# ./run.sh 0.8 6 12
# ./run.sh 0.9 7 12


for alpha in $(seq 0.1 0.1 1.0); do echo $alpha $(for seed in $(seq 0 9); do f=mislabel0.6_alpha"$alpha"_seed"$seed".log; cat $f | grep val_acc | awk '{print $NF}' | sort -nk1,1 | tail | awk '{sum+=$1;cnt+=1}END{print sum/cnt}' ; done | awk '{sum+=$1;cnt+=1}END{printf("%.6f\n", sum/cnt)}') ; done