# E2E Tests with Playwright

This directory contains end-to-end tests for the Spark Resource Optimizer dashboard using Playwright.

## Test Structure

```
e2e/
├── dashboard.spec.ts        # Tests for the main dashboard page
├── analytics.spec.ts        # Tests for the analytics page
├── recommendations.spec.ts  # Tests for the recommendations page
└── README.md               # This file
```

## Prerequisites

Before running the tests, you need to:

1. **Install Playwright browsers** (one-time setup):
   ```bash
   cd web-ui-dashboard
   pnpm exec playwright install
   ```

2. **Start the backend server** (in a separate terminal):
   ```bash
   # From project root
   python -m spark_optimizer.api.server
   ```

## Running Tests

### Run all tests
```bash
cd web-ui-dashboard
pnpm e2e
```

### Run tests in UI mode (interactive)
```bash
pnpm e2e:ui
```

### Run tests in headed mode (see browser)
```bash
pnpm e2e:headed
```

### Run tests for specific browser
```bash
pnpm e2e:chromium    # Chromium only
pnpm e2e --project=firefox   # Firefox only
pnpm e2e --project=webkit    # WebKit only
```

### Run specific test file
```bash
pnpm e2e e2e/dashboard.spec.ts
pnpm e2e e2e/analytics.spec.ts
pnpm e2e e2e/recommendations.spec.ts
```

### Run tests on mobile viewports
```bash
pnpm e2e --project="Mobile Chrome"
pnpm e2e --project="Mobile Safari"
```

### View test report
```bash
pnpm e2e:report
```

## Test Coverage

### Dashboard Tests (`dashboard.spec.ts`)
- ✅ Page title and description display
- ✅ Statistics cards (Total Jobs, Successful, Failed, Avg Duration)
- ✅ Auto-refresh toggle functionality
- ✅ Last updated timestamp
- ✅ Recent jobs table with proper columns
- ✅ Loading state
- ✅ Responsive design (mobile menu)
- ✅ Navigation to other pages

### Analytics Tests (`analytics.spec.ts`)
- ✅ Page title and description
- ✅ Job duration trends chart
- ✅ Success rate trends chart
- ✅ Job status distribution chart
- ✅ Resource utilization chart
- ✅ Cost trends chart
- ✅ Data processing volume chart
- ✅ Loading and error states
- ✅ Responsive charts

### Recommendations Tests (`recommendations.spec.ts`)
- ✅ Form display and validation
- ✅ Default form values
- ✅ Required field validation
- ✅ Recommendation request submission
- ✅ Results display with confidence score
- ✅ Resource allocation chart
- ✅ Similar jobs visualization
- ✅ Cost-performance indicator
- ✅ Copy to clipboard functionality
- ✅ Form reset
- ✅ Error handling
- ✅ Loading state

## CI/CD Integration

Tests run automatically in CI via GitHub Actions on:
- Push to `main` or `feature/**` branches
- Pull requests to `main`

The CI workflow:
1. Sets up Python and Node.js environments
2. Installs dependencies
3. Creates and seeds a test database
4. Starts the backend server
5. Runs E2E tests in parallel across multiple browsers
6. Uploads test reports and videos on failure

## Writing New Tests

### Basic Test Structure

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/your-route');
  });

  test('should do something', async ({ page }) => {
    // Your test code
    await expect(page.getByText('Expected Text')).toBeVisible();
  });
});
```

### Best Practices

1. **Use semantic selectors**: Prefer `getByRole`, `getByLabel`, `getByText` over CSS selectors
2. **Wait for elements**: Use `toBeVisible()` with timeouts for async content
3. **Mock API calls**: Use `page.route()` to intercept and mock API responses
4. **Test error states**: Include tests for loading, error, and empty states
5. **Keep tests isolated**: Each test should be independent
6. **Use descriptive names**: Test names should clearly describe what is being tested

### Mocking API Responses

```typescript
test('should handle API response', async ({ page }) => {
  await page.route('**/api/v1/endpoint', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ /* your mock data */ })
    });
  });

  // Your test code
});
```

## Debugging Tests

### Debug a specific test
```bash
pnpm exec playwright test --debug e2e/dashboard.spec.ts
```

### Generate test code interactively
```bash
pnpm exec playwright codegen http://localhost:4200
```

### Show trace viewer
```bash
pnpm exec playwright show-trace path/to/trace.zip
```

## Configuration

Test configuration is in `playwright.config.ts`:
- Base URL: `http://localhost:4200`
- Test timeout: 30 seconds (default)
- Retries: 0 locally, 2 in CI
- Workers: Parallel locally, 1 in CI
- Screenshots: Only on failure
- Traces: On first retry

## Troubleshooting

### Backend not running
If tests fail with connection errors, ensure the backend is running:
```bash
python -m spark_optimizer.api.server
```

### Port already in use
If port 4200 or 8080 is in use:
```bash
# Find and kill process on port 4200
lsof -ti:4200 | xargs kill

# Find and kill process on port 8080
lsof -ti:8080 | xargs kill
```

### Browsers not installed
If you see "Executable doesn't exist" errors:
```bash
pnpm exec playwright install
```

### Test database issues
If tests fail due to missing data:
```bash
# Delete and recreate test database
rm data/test_spark_optimizer.db
python -c "from spark_optimizer.storage.database import Database; db = Database('sqlite:///data/test_spark_optimizer.db'); db.create_tables()"
```

## Resources

- [Playwright Documentation](https://playwright.dev)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- [Playwright API Reference](https://playwright.dev/docs/api/class-playwright)
