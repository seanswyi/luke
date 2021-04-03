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
