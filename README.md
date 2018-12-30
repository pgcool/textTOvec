# textTOvec
ICLR 2019 paper: "textTOvec: DEEP CONTEXTUALIZED NEURAL AUTOREGRESSIVE TOPIC MODELS OF LANGUAGE WITH DISTRIBUTED COMPOSITIONAL PRIOR"


## About
This code consists of the implementations for the model proposed in the paper published at ICLR 2019: "textTOvec: DEEP CONTEXTUALIZED NEURAL AUTOREGRESSIVE TOPIC MODELS OF LANGUAGE WITH DISTRIBUTED COMPOSITIONAL PRIOR".

Paper: https://arxiv.org/pdf/1810.03947.pdf


## Requirements
Requires Python 3 (tested with `3.6.1`). The remaining dependencies can then be installed via:

        $ pip install -r requirements.txt
        $ python -c "import nltk; nltk.download('all')"


## Data format

**Datasets**: A directory containing CSV files. There is expected to be 1 CSV file per set or collection, with separate sets for training, validation and test. The CSV files in the directory must be named accordingly for DocNADE model: `training_docnade.csv`, `validation_docnade.csv`, `test_docnade.csv`. The CSV files in the directory must be named accordingly for lstm portion of the model `ctx-DocNADE(e)`: `training_lstm.csv`, `validation_lstm.csv`, `test_lstm.csv`. For this task, each CSV file (prior to preprocessing) consists of 2 string fields with a comma delimiter - the first is the label and the second is the document body.


**Vocabulary files**: A plain text file, with 1 vocabulary token per line (note that this must be created in advance, we do not provide a script for creating vocabularies), each for DocNADE and lstm portions.


**mapping_dict.pkl**: A dictionary that maps the indices of words in DocNADE to indices of words in LSTM data format (i.e., CSVs).


## How to use: Train baseline DocNADE model 

The script train_DATANAME_docnade_PPL.sh or train_DATANAME_docnade_IR.sh invokes train_model.py to train the baseline DocNADE model, compute PPL/IR and save it in a repository. It will also log all the information with the PPL and IR models in the seperate directories. Here's how to use the script:

        $ ./train_20NSshort_docnade_PPL.sh
		# to compute PPL
		
		$ ./train_20NSshort_docnade_IR.sh
		# to compute IR
		
		--dataset				is the path to the input dataset. 
		--docnadeVocab 			is the path to the vocabulary of the input dataset in DocNADE portion. 
		--model 				is the path to the save the best model.
		--initialize-docnade 	Init DocNADE weights. False for DocNADE; True/False for ctx-DocNADE(e) model 
		--bidirectional 		True, if bidirectional settings reqquired in DocNADE. Default: False
		--activation 			*sigmoid* for PPL and *tanh* for IR computations. 
		--learning-rate 		0.001 
		--batch-size 			training batch szie, for instance, 100 
		--num-steps 			the number of training steps  
		--log-every 		 
		--validation-bs 		validation batch size, set to 1
		--test-bs 				test batch size, set to 1
		--validation-ppl-freq 	computate PPL of validation set at this frequency  
		--validation-ir-freq	computate IR of validation set at this frequency    
		--test-ir-freq  		computate IR of test set at this frequency  
		--test-ppl-freq  		computate PPL of test set at this frequency  
		--num-classes 			number of class labels; not used.
		--patience 				stopping criteria on validation scores
		--supervised 			If training in supervised setting. Set to False.
		--hidden-size 			The number of hidden units in a hiden vector 
		--combination-type 		The mode of combining hidden vectors from DocNADE and LSTM portions. Set to 'sum'  
		--vocab-size 			Voabulary size in DocNADE portion. 
		--deep 					True, if additional layers on both DocNADE and LSTM portions. Set False for non-deep versions. 
		--deep-hidden-sizes 	List of hidden sizes. For instance, for a two layered network, set: 200 200. Used, if deep = True
		--trainfile 			is path to training text file. (required in case of topic coherence)
		--valfile 				is path to validation text file. (required in case of topic coherence)
		--testfile 				is path to testing text file. (required in case of topic coherence)
		--reload 				True, if reloading of a model required. Set to False otherwise.  
		--reload-model-dir 		Path to the model to reload. Used only if reload set to True. 
		
		

        *** TO DO ***: Improve Documentation. 

		
		
## How to use: Train ctx-DocNADE or ctx-DocNADEe model 

The script train_20NSshort_docnade_lstm_PPL.sh or train_20NSshort_docnade_lstm_IR.sh invokes train_model_lstm.py to train the ctx-DocNADE or ctx-DocNADEe model, compute PPL/IR and save it in a repository. It will also log all the information with the PPL and IR models in the seperate directories. Here's how to use the script:

        $ ./train_20NSshort_docnade_lstm_PPL.sh
		# to compute PPL using ctx-DocNADE or crx-DocNADEe model i.e., textTOvec models
		
		$ ./train_20NSshort_docnade_lstm_IR.sh
		# to compute IR using ctx-DocNADE or crx-DocNADEe model i.e., textTOvec models
		
		
		--dataset				is the path to the input dataset. 
		--mapping-dict			is the path to mapping_dict.pkl file.
		--rnnVocab				is the path to the vocabulary of the input dataset in LSTM portion.
		--docnadeVocab 			is the path to the vocabulary of the input dataset in DocNADE portion. 
		--model 				is the path to the save the best model.
		--initialize-docnade 	Init DocNADE weights. False for DocNADE; True/False for ctx-DocNADE(e) model 
		--bidirectional 		True, if bidirectional settings reqquired in DocNADE. Default: False
		--activation 			sigmoid for PPL and tanh for IR computations. 
		--learning-rate 		0.001 
		--batch-size 			training batch szie, for instance, 100 
		--num-steps 			the number of training steps  
		--log-every 		 
		--validation-bs 		validation batch size, set to 1
		--test-bs 				test batch size, set to 1
		--validation-ppl-freq 	computate PPL of validation set at this frequency  
		--validation-ir-freq	computate IR of validation set at this frequency    
		--test-ir-freq  		computate IR of test set at this frequency  
		--test-ppl-freq  		computate PPL of test set at this frequency  
		--num-classes 			number of class labels; not used.
		--patience 				stopping criteria on validation scores
		--supervised 			If training in supervised setting. Set to False.
		--hidden-size 			The number of hidden units in a hiden vector 
		--combination-type 		The mode of combining hidden vectors from DocNADE and LSTM portions. Set to 'sum'  
		--vocab-size 			Voabulary size in DocNADE portion. 
		--deep 					True, if additional layers on both DocNADE and LSTM portions. Set False for non-deep versions. 
		--deep-hidden-sizes 	List of hidden sizes. For instance, for a two layered network, set: 200 200. Used, if deep = True
		--use-docnade-for-ir 	True, to log IR due to DocNADE portion *only* in the ctx-DocNADE ctx-DocNADEe model 
		--use-lstm-for-ir 		True, to log IR due to LSTM portion *only* in the ctx-DocNADE ctx-DocNADEe model 
		--use-combination-for-ir True, to log IR due to DocNADE+LSTM portion together in the ctx-DocNADE ctx-DocNADEe model 
		--initialize-rnn 		Init LSTM with Glove embeddings, i.e., ctx-DocNADEe 
		--update-docnade-w 		True, to update weights in DocNADE portion.  
		--update-rnn-w 			False, to not update embeddings in LSTM portion. 
		--lambda-hidden-lstm 	mixture weight, lambda in [0.0-1.0]
		--trainfile 			is path to training text file. (required in case of topic coherence)
		--valfile 				is path to validation text file. (required in case of topic coherence)
		--testfile 				is path to testing text file. (required in case of topic coherence)
		--reload 				True, if reloading of a model required. Set to False otherwise.  
		--reload-model-dir 		Path to the model to reload. Used only if reload set to True. 
		--reload-docnade-embeddings True, to init DocNADE portion of the ctx-DocNADE from a pretrained DocNADE model
		--docnade-embeddings-path Path to topic-embedding matrix W from a pre-trained DocNADE model
		
		

        *** TO DO ***: Improve Documentation. 


		
## Directory structure for results and datasets

# Contains dataset folders
Datasets directory:             ./datasets/

# Contains GloVe pretrained embeddings
Pre-trained embeddings dir:     /home/usr/resources/pretrained_embeddings/

# Contains results of training
Results directory:              ./model/

Saved logs model dir:           ./model/MODELNAME/logs/



NOTE: We will improve the readme and code documentation soon. 