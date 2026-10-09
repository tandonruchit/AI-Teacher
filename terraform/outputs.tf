output "cloud_run_service_url" {
  description = "Public HTTPS URL of the deployed ArgonLearn Cloud Run service."
  value       = google_cloud_run_v2_service.argon_learn.uri
}

output "artifact_registry_repository_url" {
  description = "Artifact Registry Docker repository URI for container pushes."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.argon_repo.repository_id}"
}

output "firestore_database_name" {
  description = "Provisioned Google Cloud Firestore database identifier."
  value       = google_firestore_database.learner_memory.name
}

output "service_account_email" {
  description = "Least-privilege service account email bound to the Cloud Run service."
  value       = google_service_account.argon_runtime_sa.email
}

output "secret_manager_secret_id" {
  description = "Fully qualified Secret Manager resource ID for the Gemini API key."
  value       = google_secret_manager_secret.gemini_api_key.id
}
