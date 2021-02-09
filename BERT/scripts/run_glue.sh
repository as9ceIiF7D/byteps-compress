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

set -e

repo_name=${1:-"checkpoints-lans-2k"}
task_name=${2:-"MRPC"}
learning_rate=${3:-"2e-5"}
data_name=${4:-$task_name}
echo "lr=" $learning_rate
seed=${5:-"2"}


WORKSPACE=$HOME/repos/DeepLearningExamples/PyTorch/LanguageModeling/BERT
CODEDIR=$WORKSPACE
RESULTS_DIR=$CODEDIR/results
init_checkpoint=${17:-$RESULTS_DIR/$repo_name/ckpt_250000.pt}

BERT_PREP_WORKING_DIR=$HOME/datasets
data_dir=${6:-"$BERT_PREP_WORKING_DIR/glue_data/$data_name/"}
vocab_file=${7:-$WORKSPACE/vocab/vocab}
config_file=${8:-$CODEDIR/bert_base_config.json}
out_dir=${9:-$WORKSPACE/results/$data_name}
num_gpu=${10:-"8"}
batch_size=${11:-"32"}
gradient_accumulation_steps=${12:-"1"}
warmup_proportion=${13:-"0.1"}
epochs=${14:-"3"}
max_steps=${15:-"-1.0"}
precision=${16:-"fp32"}
mode=${18:-"train eval"}

mkdir -p $out_dir

if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python3 $mpi_command run_glue.py "
task_name=`echo "$task_name" | tr '[:upper:]' '[:lower:]'`
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"prediction"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"prediction"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-base-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
# CMD+="--loss_scale 256 "
CMD+="$use_fp16"

LOGFILE=$out_dir/logfile

$CMD |& tee $LOGFILE
