.PHONY: show clean debug

show:
	# Mostramos o projeto inteiro
	@grep "" -r ./src --exclude-dir=*checkpoints

debug:
	python3 -m src.main --monitor-memory
	python3 src/assets/mem_report.py mem_dumps/
	python3 src/assets/mem_report.py mem_dumps/ --objs

clean:
	cleanup "__pycache__" -r ./src