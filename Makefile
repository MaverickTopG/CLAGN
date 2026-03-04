.PHONY: validate test

validate:
	python reproduce_all.py --data_dir tests/data --benchmark_dir tests/data/benchmark --output_dir results_test --offline

test:
	pytest -q
