variable "project_id" {
  description = "The Google Cloud Project ID where ArgonLearn resources will be provisioned."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Artifact Registry, and Firestore."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment tier (development, staging, production)."
  type        = string
  default     = "production"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "environment must be one of: development, staging, production."
  }
}

variable "service_name" {
  description = "Name of the Cloud Run multi-agent service."
  type        = string
  default     = "argon-learn"
}

variable "container_image" {
  description = "Fully qualified Artifact Registry container image URI for ArgonLearn."
  type        = string
  default     = "us-central1-docker.pkg.dev/argon-learn-prod/argon-learn-repo/argon-learn:latest"
}

variable "gemini_api_key_secret_id" {
  description = "Secret Manager secret identifier storing the Gemini / Vertex AI API key."
  type        = string
  default     = "argon-learn-gemini-api-key"
}

variable "min_instances" {
  description = "Minimum warm Cloud Run instances to eliminate cold-start latency during demos."
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum autoscaling Cloud Run instances."
  type        = number
  default     = 10
}

variable "firestore_database_id" {
  description = "Firestore Native database identifier for persistent learner sessions and DAGs."
  type        = string
  default     = "(default)"
}
