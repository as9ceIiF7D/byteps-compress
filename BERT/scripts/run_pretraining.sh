#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-64}
learning_rate=${2:-"0.00125"}
precision=${3:-"fp16"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.0125"}
train_steps=${6:-225000}
save_checkpoint_steps=${7:-5000}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-1}
seed=${12:-12439}
job_name=${13:-"bert_lans_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
train_batch_size_phase2=${16:-128}
learning_rate_phase2=${17:-"0.00125"}
warmup_proportion_phase2=${18:-"0.0125"}
train_steps_phase2=${19:-25000}
gradient_accumulation_steps_phase2=${20:-4}
DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1536_large/books_wiki_en_corpus_train # change this for other datasets
BERT_PREP_WORKING_DIR=$HOME/datasets
DATA_DIR_PHASE1=${21:-$BERT_PREP_WORKING_DIR/${DATASET}/}
DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1536_large/books_wiki_en_corpus_train # change this for other datasets
DATA_DIR_PHASE2=${22:-$BERT_PREP_WORKING_DIR/${DATASET2}/}
WORKSPACE=$HOME/repos/DeepLearningExamples/PyTorch/LanguageModeling/BERT
CODEDIR=${23:-$WORKSPACE}
BERT_CONFIG=$CODEDIR/bert_base_config.json
init_checkpoint=${24:-"None"}
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints-lans-2k

clush --hostfile ~/hostfile "mkdir -p $CHECKPOINTS_DIR"

# byteps 

## path
interface=ens3
ip=$(ifconfig $interface | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
port=1234
NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

repo_path=$HOME/repos/byteps
worker_hosts=worker-hosts
server_hosts=server-hosts
pem_file=${25:-""}

## finetune params
threadpool_size=0
omp_num_threads=4
partition_bytes=4096000
min_compress_bytes=1024000
server_engine_thread=4


if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-base-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger_phase1.json "
# compression 
# CMD+=" --compressor onebit --onebit-scaling --ef vanilla"
# CMD+=" --compressor topk --k 0.001 --ef vanilla"

# byteps env
ENV=""
ENV+=" --env OMP_WAIT_POLICY:PASSIVE"
ENV+=" --env OMP_NUM_THREADS:$omp_num_threads"
ENV+=" --env BYTEPS_THREADPOOL_SIZE:$threadpool_size"
ENV+=" --env BYTEPS_MIN_COMPRESS_BYTES:$min_compress_bytes"
ENV+=" --env BYTEPS_NUMA_ON:1"
ENV+=" --env NVIDIA_VISIBLE_DEVICES:$NVIDIA_VISIBLE_DEVICES"
ENV+=" --env BYTEPS_SERVER_ENGINE_THREAD:$server_engine_thread"
ENV+=" --env BYTEPS_PARTITION_BYTES:$partition_bytes"
ENV+=" --env BYTEPS_LOG_LEVEL:INFO"
ENV+=" --env BYTEPS_FORCE_DISTRIBUTED:1"
# ENV+=" --env BYTEPS_TRACE_ON:1"
# ENV+=" --env BYTEPS_TRACE_START_STEP:100"
# ENV+=" --env BYTEPS_TRACE_END_STEP:110"
# ENV+=" --env BYTEPS_TRACE_DIR:./traces"

CMD="python3 $repo_path/launcher/dist_launcher.py -WH $worker_hosts -SH $server_hosts --scheduler-ip $ip --scheduler-port $port --interface $interface -i $pem_file --username ubuntu $ENV source ~/.zshrc; bpslaunch python3 $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

sleep 110000

set +x

echo "finished phase1"

#Start Phase2

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-base-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger_phase2.json "
# compression 
# CMD+=" --compressor onebit --onebit-scaling --ef vanilla"
# CMD+=" --compressor topk --k 0.001 --ef vanilla"

# byteps env
ENV=""
ENV+=" --env OMP_WAIT_POLICY:PASSIVE"
ENV+=" --env OMP_NUM_THREADS:$omp_num_threads"
ENV+=" --env BYTEPS_THREADPOOL_SIZE:$threadpool_size"
ENV+=" --env BYTEPS_MIN_COMPRESS_BYTES:$min_compress_bytes"
ENV+=" --env BYTEPS_NUMA_ON:1"
ENV+=" --env NVIDIA_VISIBLE_DEVICES:$NVIDIA_VISIBLE_DEVICES"
ENV+=" --env BYTEPS_SERVER_ENGINE_THREAD:$server_engine_thread"
ENV+=" --env BYTEPS_PARTITION_BYTES:$partition_bytes"
ENV+=" --env BYTEPS_LOG_LEVEL:INFO"
ENV+=" --env BYTEPS_FORCE_DISTRIBUTED:1"

CMD="python3 $repo_path/launcher/dist_launcher.py -WH $worker_hosts -SH $server_hosts --scheduler-ip $ip --scheduler-port $port --interface $interface -i $pem_file --username ubuntu $ENV source ~/.zshrc; bpslaunch python3 $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished phase2"
