{{/*
Expand the name of the chart.
*/}}
{{- define "spark-resource-optimizer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "spark-resource-optimizer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "spark-resource-optimizer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "spark-resource-optimizer.labels" -}}
helm.sh/chart: {{ include "spark-resource-optimizer.chart" . }}
{{ include "spark-resource-optimizer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "spark-resource-optimizer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "spark-resource-optimizer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "spark-resource-optimizer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "spark-resource-optimizer.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the database URL
*/}}
{{- define "spark-resource-optimizer.databaseUrl" -}}
{{- if .Values.database.external }}
{{- printf "postgresql://%s:%s@%s:%d/%s?sslmode=%s" .Values.database.username .Values.database.password .Values.database.host (int .Values.database.port) .Values.database.name .Values.database.sslMode }}
{{- else if .Values.postgresql.enabled }}
{{- $host := printf "%s-postgresql" .Release.Name }}
{{- $password := .Values.postgresql.auth.password }}
{{- printf "postgresql://%s:%s@%s:5432/%s" .Values.postgresql.auth.username $password $host .Values.postgresql.auth.database }}
{{- else }}
{{- "sqlite:///spark_optimizer.db" }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "spark-resource-optimizer.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{/*
Return the secret name for database credentials
*/}}
{{- define "spark-resource-optimizer.databaseSecretName" -}}
{{- printf "%s-db-secret" (include "spark-resource-optimizer.fullname" .) }}
{{- end }}

{{/*
Return the secret name for JWT
*/}}
{{- define "spark-resource-optimizer.jwtSecretName" -}}
{{- printf "%s-jwt-secret" (include "spark-resource-optimizer.fullname" .) }}
{{- end }}

{{/*
Return the configmap name
*/}}
{{- define "spark-resource-optimizer.configMapName" -}}
{{- printf "%s-config" (include "spark-resource-optimizer.fullname" .) }}
{{- end }}

{{/*
Worker labels
*/}}
{{- define "spark-resource-optimizer.workerLabels" -}}
helm.sh/chart: {{ include "spark-resource-optimizer.chart" . }}
{{ include "spark-resource-optimizer.workerSelectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "spark-resource-optimizer.workerSelectorLabels" -}}
app.kubernetes.io/name: {{ include "spark-resource-optimizer.name" . }}-worker
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: worker
{{- end }}
