PYTHONPATH=/home/ubuntu/topic_modeling/static/textTOvec
export MKL_THREADING_LAYER=GNU
export THEANO_FLAGS='floatX=float64,device=cpu,exception_verbosity=high'
export OMP_NUM_THREADS=3
python train_model_lstm.py --dataset ./datasets/20NSshort --mapping-dict ./datasets/20NSshort/mapping_dict.pkl --rnnVocab ./datasets/20NSshort/vocab_lstm.vocab --docnadeVocab ./datasets/20NSshort/vocab_docnade.vocab --model ./model/20NSshort --learning-rate 0.001 --batch-size 100 --validation-bs 1 --test-bs 1 --log-every 2 --num-steps 20000000 --patience 1000 --validation-ppl-freq 1000000 --validation-ir-freq 12 --test-ppl-freq 1000000 --test-ir-freq 12 --num-classes 20 --supervised False --hidden-size 200 --deep-hidden-sizes 200 200 --activation tanh --use-docnade-for-ir True --use-lstm-for-ir True --use-combination-for-ir True --combination-type sum --initialize-docnade True --initialize-rnn True --update-docnade-w True --update-rnn-w False --vocab-size 1448 --include-lstm-loss False --common-space False --deep False --multi-label False --reload False --reload-train False --docnade-loss-weight 1.0 --lstm-loss-weight 0.0 --lambda-hidden-lstm 0.5 --reload-docnade-embeddings True --reload-model-dir 20NSshort_DocNADE_act_tanh_hidden_200_vocab_1448_lr_0.001_proj_False_deep_False_19_9_2018 --docnade-embeddings-path ./docnade_embeddings_ir_full_vocab/20NSshort