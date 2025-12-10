# Spark Resource Optimizer Helm Chart

This Helm chart deploys Spark Resource Optimizer on a Kubernetes cluster.

## Prerequisites

- Kubernetes 1.23+
- Helm 3.8+
- PV provisioner support in the underlying infrastructure (if persistence is enabled)

## Installation

### Add the repository (if published)

```bash
helm repo add gridatek https://charts.gridatek.com
helm repo update
```

### Install from local chart

```bash
# Navigate to the helm directory
cd helm

# Install with default values
helm install spark-optimizer ./spark-resource-optimizer

# Install with custom values
helm install spark-optimizer ./spark-resource-optimizer -f my-values.yaml

# Install in a specific namespace
helm install spark-optimizer ./spark-resource-optimizer -n spark-system --create-namespace
```

## Configuration

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of API server replicas | `2` |
| `image.repository` | Image repository | `gridatek/spark-resource-optimizer` |
| `image.tag` | Image tag | `""` (uses appVersion) |
| `config.features.auth` | Enable authentication | `true` |
| `config.features.monitoring` | Enable real-time monitoring | `true` |
| `config.features.tuning` | Enable auto-tuning | `true` |
| `config.features.costModeling` | Enable cost modeling | `true` |

### Database Configuration

#### Using Built-in PostgreSQL (default)

```yaml
postgresql:
  enabled: true
  auth:
    username: spark_optimizer
    password: "your-secure-password"
    database: spark_optimizer
```

#### Using External Database

```yaml
database:
  external: true
  host: "your-postgres-host"
  port: 5432
  name: spark_optimizer
  username: spark_optimizer
  password: "your-password"
  sslMode: require

postgresql:
  enabled: false
```

### Authentication Configuration

```yaml
config:
  features:
    auth: true

auth:
  jwt:
    secretKey: ""  # Auto-generated if empty
    accessTokenExpireMinutes: 30
    refreshTokenExpireDays: 7
  initialAdmin:
    enabled: true
    username: admin
    email: admin@example.com
    password: ""  # Auto-generated if empty
```

### Ingress Configuration

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: spark-optimizer.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: spark-optimizer-tls
      hosts:
        - spark-optimizer.example.com
```

### Resource Limits

```yaml
resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi
```

### Autoscaling

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## Upgrading

```bash
helm upgrade spark-optimizer ./spark-resource-optimizer -f my-values.yaml
```

## Uninstalling

```bash
helm uninstall spark-optimizer
```

## Production Considerations

### Security

1. **Set strong passwords** for database and JWT secret
2. **Enable TLS** via ingress
3. **Use network policies** to restrict traffic
4. **Run as non-root** (enabled by default)

### High Availability

1. Set `replicaCount >= 2`
2. Enable `autoscaling`
3. Configure `podDisruptionBudget`
4. Use external PostgreSQL with replication

### Monitoring

1. Enable Prometheus ServiceMonitor
2. Configure alerting rules
3. Use Grafana dashboards

## Troubleshooting

### Check pod status

```bash
kubectl get pods -l app.kubernetes.io/name=spark-resource-optimizer
```

### View logs

```bash
kubectl logs -l app.kubernetes.io/name=spark-resource-optimizer -f
```

### Check configuration

```bash
kubectl get configmap spark-optimizer-config -o yaml
```

### Get initial admin password

```bash
kubectl get secret spark-optimizer-jwt-secret -o jsonpath='{.data.INITIAL_ADMIN_PASSWORD}' | base64 -d
```
