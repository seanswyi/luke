run_luke_tacred_no_train:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		relation-classification run \
		--data-dir=/hdd1/seokwon/data/TACRED/data/json \
		--checkpoint-file=/hdd1/seokwon/research/luke/pytorch_model.bin \
		--no-train

run_luke_tacred:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		relation-classification run \
		--data-dir=/hdd1/seokwon/data/TACRED/data/json \
		--train-batch-size=4 \
		--gradient-accumulation-steps=8 \
		--learning-rate=1e-5 \
		--num-train-epochs=5

run_luke_docred:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		relation-classification run \
		--data-dir=/hdd1/seokwon/data/DocRED/TACRED-style \
		--setting document \
		--train-batch-size=2 \
		--eval-batch-size=4 \
		--gradient-accumulation-steps=8 \
		--learning-rate=1e-5 \
		--num-train-epochs=5

run_luke_docred_multi_gpu:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		relation-classification run \
		--data-dir=/hdd1/seokwon/data/DocRED/TACRED-style \
		--setting document \
		--train-batch-size=2 \
		--gradient-accumulation-steps=8 \
		--learning-rate=1e-5 \
		--num-train-epochs=1 \
		--multi-gpu

run_docred:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		docred run \
		--data-dir=/hdd1/seokwon/data/DocRED/original \
		--train-batch-size=2 \
		--gradient-accumulation-steps=8 \
		--learning-rate=${lr} \
		--num-train-epochs=${epochs} \
		--classifier=${classifier}

run_docred_atloss:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		docred run \
		--data-dir=/hdd1/seokwon/data/DocRED/original \
		--train-batch-size=4 \
		--gradient-accumulation-steps=8 \
		--learning-rate=${lr} \
		--num-train-epochs=${epochs} \
		--classifier=${classifier} \
		--at_loss

run_docred_lop:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		docred run \
		--data-dir=/hdd1/seokwon/data/DocRED/original \
		--train-batch-size=4 \
		--gradient-accumulation-steps=8 \
		--learning-rate=${lr} \
		--num-train-epochs=${epochs} \
		--classifier=${classifier} \
		--lop

run_docred_debug:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		docred run \
		--data-dir=/hdd1/seokwon/data/DocRED/original \
		--train-batch-size=1 \
		--gradient-accumulation-steps=1 \
		--learning-rate=1e-5 \
		--num-train-epochs=5

run_docred_multi_gpu:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		docred run \
		--data-dir=/hdd1/seokwon/data/DocRED/original \
		--train-batch-size=4 \
		--gradient-accumulation-steps=8 \
		--learning-rate=1e-5 \
		--num-train-epochs=1 \
		--multi-gpu

run_luke_docred_single_batch:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		relation-classification run \
		--data-dir=/hdd1/seokwon/data/DocRED/TACRED-style \
		--setting document \
		--train-batch-size=1 \
		--gradient-accumulation-steps=1 \
		--learning-rate=1e-5 \
		--num-train-epochs=1

debug_luke_docred:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		relation-classification run \
		--data-dir=/hdd1/seokwon/data/DocRED/TACRED-style \
		--setting document \
		--train-batch-size=4 \
		--gradient-accumulation-steps=8 \
		--learning-rate=1e-5 \
		--num-train-epochs=5 \
		--debug

run_luke_conll:
	python -m examples.cli \
		--model-file=/hdd1/seokwon/research/luke/luke_large_500k.tar.gz \
		--output-dir=/hdd1/seokwon/research/luke \
		ner run \
		--data-dir=/hdd1/seokwon/data/CoNLL-2003 \
		--train-batch-size=2 \
		--gradient-accumulation-steps=2 \
		--learning-rate=1e-5 \
		--num-train-epochs=5
