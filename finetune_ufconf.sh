[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${update_freq}" ] && update_freq=1
[ -z "${total_step}" ] && total_step=10000
[ -z "${warmup_step}" ] && warmup_step=500
[ -z "${decay_step}" ] && decay_step=10000
[ -z "${decay_ratio}" ] && decay_ratio=1.0
[ -z "${sd_prob}" ] && sd_prob=0.5
[ -z "${lr}" ] && lr=5e-4
[ -z "${seed}" ] && seed=31
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${load_from_ema}" ] && load_from_ema="--load-from-ema"
[ -z "${disable_sd}" ] && disable_sd=""  # "--disable-sd"
[ -z "${use_multimer}" ] && use_multimer="" # "--use-multimer"
[ -z "${use_cluster_dataset}" ] && use_cluster_dataset=""  # "--use-cluster-dataset"


[ -z "${valid_step}" ] && valid_step=500
[ -z "${save_step}" ] && save_step=1000
[ -z "${log_interval}" ] && log_interval=10

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $1
echo "save_dir" $2
echo "decay_step" $decay_step
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "lr" $lr
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "data_folder:"
ls $1
echo "create folder for save"
mkdir -p $2
echo "start training"

OPTION=""
if [ -f "$2/checkpoint_last.pt" ]; then
    echo "ckp exists."
else
  echo "finetuning from inital training..."
  OPTION=" --finetune-from-model $3 $load_from_ema "
fi
model_name=$4

tmp_dir=`mktemp -d`

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       $(which unicore-train) $1 --user-dir ufconf $use_multimer $disable_sd $use_cluster_dataset \
       --num-workers 8 --ddp-backend=no_c10d \
       --task ufconf --loss ufconf --arch ufconf  --sd-prob $sd_prob  \
       --valid-subset "eval_train,eval_init,eval_half,eval_last"  \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0  --per-sample-clip-norm 0.1 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay --batch-size 1 \
       --update-freq $update_freq --seed $seed  --tensorboard-logdir $2/tsb/ \
       --max-update $total_step --max-epoch 1 --log-interval $log_interval --log-format simple \
       --save-interval-updates $save_step --validate-interval-updates $valid_step --keep-interval-updates 20 --no-epoch-checkpoints  \
       --save-dir $2 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --bf16 --ema-decay -1 --data-buffer-size 32 --bf16-sr --model-name $model_name $OPTION --fixed-validation-seed 38

rm -rf $tmp_dir
