terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.40"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ------------------------------------------------------------------------------
# 1. Enable Required Google Cloud APIs
# ------------------------------------------------------------------------------
locals {
  required_apis = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "firestore.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "cloudtasks.googleapis.com",
    "aiplatform.googleapis.com",
    "logging.googleapis.com",
  ])
}

resource "google_project_service" "enabled_apis" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ------------------------------------------------------------------------------
# 2. Artifact Registry for Container Images
# ------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "argon_repo" {
  location      = var.region
  repository_id = "${var.service_name}-repo"
  description   = "Docker container registry for ArgonLearn ADK multi-agent application"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-recent-releases"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# 3. Google Cloud Firestore (Persistent Learner Session & Concept Mastery DAG)
# ------------------------------------------------------------------------------
resource "google_firestore_database" "learner_memory" {
  project                     = var.project_id
  name                        = var.firestore_database_id
  location_id                 = var.region
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"
  point_in_time_recovery_enablement = "POINT_IN_TIME_RECOVERY_ENABLED"

  depends_on = [google_project_service.enabled_apis]
}

resource "google_firestore_index" "learner_concept_dag_index" {
  project    = var.project_id
  database   = google_firestore_database.learner_memory.name
  collection = "learner_knowledge_nodes"

  fields {
    field_path = "learner_id"
    order      = "ASCENDING"
  }

  fields {
    field_path = "mastery_score"
    order      = "DESCENDING"
  }

  fields {
    field_path = "updated_at"
    order      = "DESCENDING"
  }
}

# ------------------------------------------------------------------------------
# 4. Secret Manager for Gemini API Key & Credentials
# ------------------------------------------------------------------------------
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = var.gemini_api_key_secret_id

  labels = {
    app         = var.service_name
    environment = var.environment
    managed_by  = "terraform"
  }

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# 5. Dedicated Least-Privilege Service Account & IAM Bindings
# ------------------------------------------------------------------------------
resource "google_service_account" "argon_runtime_sa" {
  account_id   = "${var.service_name}-runtime-sa"
  display_name = "ArgonLearn Multi-Agent Cloud Run Runtime Service Account"
  description  = "Least-privilege identity for ADK agents, Firestore DAG memory, Cloud Trace, and Secret Manager"
}

resource "google_secret_manager_secret_iam_member" "secret_accessor" {
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.argon_runtime_sa.email}"
}

locals {
  runtime_iam_roles = toset([
    "roles/datastore.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/dlp.user",
    "roles/cloudtasks.enqueuer",
    "roles/aiplatform.user",
  ])
}

resource "google_project_iam_member" "runtime_permissions" {
  for_each = local.runtime_iam_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.argon_runtime_sa.email}"
}

# ------------------------------------------------------------------------------
# 6. Google Cloud Run v2 Service
# ------------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "argon_learn" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.argon_runtime_sa.email
    timeout         = "300s"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2000m"
          memory = "2Gi"
        }
        cpu_idle = false
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GCP_REGION"
        value = var.region
      }

      env {
        name  = "APP_ENV"
        value = var.environment
      }

      env {
        name  = "FIRESTORE_DATABASE"
        value = google_firestore_database.learner_memory.name
      }

      env {
        name  = "GEMINI_API_KEY_SECRET_ID"
        value = var.gemini_api_key_secret_id
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        initial_delay_seconds = 3
        period_seconds        = 5
        failure_threshold     = 6
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        period_seconds    = 15
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.secret_accessor,
    google_firestore_database.learner_memory,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  name     = google_cloud_run_v2_service.argon_learn.name
  location = google_cloud_run_v2_service.argon_learn.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
