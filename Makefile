check_env:
	@if [ "$$(pyenv version-name)" = "project_dermai_assist" ]; then \
		echo "✅ Correct environment detected: project_dermai_assist"; \
		pip install -r requirements.txt; \
		pip install -r requirements_dev.txt; \
		pip install -e .; \
	else \
		echo "❌ Please activate the 'project_dermai_assist' pyenv virtualenv before running this command."; \
	fi

show_gs_models:
	@gsutil ls gs://${BUCKET_NAME}/models/*.keras
