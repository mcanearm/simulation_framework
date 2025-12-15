test:
	@echo "Running tests..."
	PYTHONPATH=. pytest tests
	@echo "Tests completed."
	
sim_example:
	@echo "Running example script..."
	PYTHONPATH=. python ./example/ridge_example.py
	@echo "Example script completed."
	
setup_env:
	@echo "Setting up the environment..."
	pip install -r requirements-dev.txt
	@echo "Environment setup completed."
	
uv_setup:
	@echo "Setting up the enviroment using uv..."
	uv sync --extra dev
	@echo "Environment setup complete."