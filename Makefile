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

reinstall_package:
	pip install -r requirements.txt && pip install -e .

run_api:
	uvicorn dermai.backend.fast:app --reload --port ${PORT}

build_local_image:
	docker build -t ${IMAGE}:${ENV} .

run_local_container:
	docker run -d --name ${CONTAINER} \
		-v ${GOOGLE_APPLICATION_CREDENTIALS}:/app/credentials.json \
		-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
		-e PORT=${PORT} \
		-p 8080:8000 \
		--env-file .env \
		${IMAGE}:${ENV}

# authenticate_artifact_registery:
# 	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# grant_permissions_to_artifact_registery:
# 	gcloud projects add-iam-policy-binding ${GCP_PROJECT} \
# 		--member="${EMAIL}" \
# 		--role="roles/artifactregistry.writer"

create_artifacts_repo:
	gcloud artifacts repositories create "${ARTIFACTSREPO}" \
		--repository-format=docker \
		--location="${GCP_REGION}" \
		--description="Our Docker Artifact Repository for LeWagon best DL project" \
		--project="${GCP_PROJECT}"

build_cloud_image:
	docker build \
		--platform linux/amd64 \
		-t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_cloud_image:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod

run_cloud_container:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--region ${GCP_REGION} \
		--service-account=${SERVICE_ACCOUNT} \
		--env-vars-file .env.yaml \
		--allow-unauthenticated

disable_cloud_service:
	gcloud run services update ${IMAGE} --min-instances=0

delete_cloud_service:
	gcloud run services delete ${IMAGE}
