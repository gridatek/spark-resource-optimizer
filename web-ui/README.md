# Spark Optimizer Web UI

Angular 21 dashboard for visualizing and optimizing Spark job configurations.

## Features

- 📊 **Dashboard**: View recent Spark jobs and statistics
- 🎯 **Recommendations**: Get optimal resource configuration recommendations
- 📈 **Real-time Updates**: Monitor job performance and metrics
- 🎨 **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

```bash
# Navigate to web-ui directory
cd web-ui/spark-optimizer-dashboard

# Install dependencies
pnpm install

# Start development server
pnpm start
```

Navigate to `http://localhost:4200/`

## Documentation

See [spark-optimizer-dashboard/README.md](spark-optimizer-dashboard/README.md) for detailed documentation.

## Project Structure

- `spark-optimizer-dashboard/` - Angular 21 application
  - `src/app/components/` - UI components
  - `src/app/services/` - API services
  - `src/app/models/` - TypeScript interfaces
  - `src/environments/` - Environment configurations

## Backend API

The UI requires the backend API running on `http://localhost:8080`

```bash
# From project root
spark-optimizer serve --port 8080
```

## Development

See the main [README](spark-optimizer-dashboard/README.md) for:
- Installation instructions
- Development workflow
- Building for production
- Deployment options

## License

MIT License - see LICENSE file for details
