all:
	python3 -m src.main

debug:
	python3 -m src.main --monitor-memory
	python3 src/assets/mem_report.py mem_dumps/
	python3 src/assets/mem_report.py mem_dumps/ --objs