import { test, expect } from '@playwright/test';

test.describe('Recommendations Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/recommendations');
  });

  test('should display recommendations title and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Get Resource Recommendations' })).toBeVisible();
    await expect(page.getByText('Get optimal Spark configuration recommendations for your job')).toBeVisible();
  });

  test('should display recommendation form', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Job Requirements' })).toBeVisible();

    // Check form fields
    await expect(page.getByLabel(/Input Data Size \(GB\)/i)).toBeVisible();
    await expect(page.getByLabel(/Job Type/i)).toBeVisible();
    await expect(page.getByLabel(/Application Name/i)).toBeVisible();
    await expect(page.getByLabel(/Recommendation Method/i)).toBeVisible();

    // Check buttons
    await expect(page.getByRole('button', { name: /Get Recommendation/i })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reset' })).toBeVisible();
  });

  test('should have default form values', async ({ page }) => {
    const inputSizeField = page.getByLabel(/Input Data Size \(GB\)/i);
    await expect(inputSizeField).toHaveValue('10');
  });

  test('should validate required fields', async ({ page }) => {
    // Clear the input size field
    const inputSizeField = page.getByLabel(/Input Data Size \(GB\)/i);
    await inputSizeField.clear();

    // Try to submit
    const submitButton = page.getByRole('button', { name: /Get Recommendation/i });
    await expect(submitButton).toBeDisabled();
  });

  test('should submit recommendation request', async ({ page }) => {
    // Fill in the form
    await page.getByLabel(/Input Data Size \(GB\)/i).fill('50');
    await page.selectOption('select[name="jobType"]', 'ml');
    await page.getByLabel(/Application Name/i).fill('test-ml-job');
    await page.selectOption('select[name="method"]', 'hybrid');

    // Mock API response
    await page.route('**/api/v1/recommend', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          predicted_metrics: {
            duration_minutes: 15,
            cost_usd: 2.5
          },
          confidence: 0.85,
          method: 'hybrid',
          metadata: {
            similar_jobs: [
              { app_id: 'app-001', similarity: 0.92 },
              { app_id: 'app-002', similarity: 0.88 }
            ]
          }
        })
      });
    });

    // Submit form
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await expect(page.getByRole('heading', { name: 'Recommended Configuration' })).toBeVisible({ timeout: 10000 });
  });

  test('should display recommendation results', async ({ page }) => {
    // Mock API response
    await page.route('**/api/v1/recommend', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          predicted_metrics: {
            duration_minutes: 15,
            cost_usd: 2.5
          },
          confidence: 0.85,
          method: 'hybrid'
        })
      });
    });

    // Submit with default values
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Verify results display
    await expect(page.getByRole('heading', { name: 'Recommended Configuration' })).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Confidence:')).toBeVisible();
    await expect(page.getByText('85%')).toBeVisible();

    // Check configuration cards
    await expect(page.getByText('Executor Cores')).toBeVisible();
    await expect(page.getByText('Executor Memory')).toBeVisible();
    await expect(page.getByText('Number of Executors')).toBeVisible();
    await expect(page.getByText('Driver Memory')).toBeVisible();

    // Check predicted metrics
    await expect(page.getByText('Predicted Performance')).toBeVisible();
    await expect(page.getByText('Duration:')).toBeVisible();
    await expect(page.getByText('Estimated Cost:')).toBeVisible();

    // Check spark command
    await expect(page.getByRole('heading', { name: 'Spark Submit Command' })).toBeVisible();
    await expect(page.getByText(/spark-submit/)).toBeVisible();
  });

  test('should display resource allocation chart', async ({ page }) => {
    // Mock API response
    await page.route('**/api/v1/recommend', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          confidence: 0.85,
          method: 'hybrid'
        })
      });
    });

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for chart to render
    await page.waitForSelector('canvas', { timeout: 10000 });
    await expect(page.getByRole('heading', { name: 'Resource Allocation' })).toBeVisible();
  });

  test('should display similar jobs visualization', async ({ page }) => {
    // Mock API response with similar jobs
    await page.route('**/api/v1/recommend', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          confidence: 0.85,
          method: 'similarity',
          metadata: {
            similar_jobs: [
              { app_id: 'app-001', similarity: 0.92 },
              { app_id: 'app-002', similarity: 0.88 },
              { app_id: 'app-003', similarity: 0.85 }
            ]
          }
        })
      });
    });

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Check similar jobs section
    await expect(page.getByRole('heading', { name: 'Similar Jobs Analysis' })).toBeVisible({ timeout: 10000 });
    await expect(page.getByText(/This recommendation is based on \d+ similar historical jobs/)).toBeVisible();
    await expect(page.getByText('Similarity:')).toBeVisible();
  });

  test('should display cost-performance indicator', async ({ page }) => {
    // Mock API response with predicted metrics
    await page.route('**/api/v1/recommend', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          predicted_metrics: {
            duration_minutes: 15,
            cost_usd: 2.5
          },
          confidence: 0.85,
          method: 'hybrid'
        })
      });
    });

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Check cost-performance indicator
    await expect(page.getByRole('heading', { name: 'Cost-Performance Indicator' })).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Cost Efficient')).toBeVisible();
    await expect(page.getByText('High Performance')).toBeVisible();
  });

  test('should copy spark command to clipboard', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    // Mock API response
    await page.route('**/api/v1/recommend', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          confidence: 0.85,
          method: 'hybrid'
        })
      });
    });

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results and click copy button
    const copyButton = page.getByRole('button', { name: /Copy Command/i });
    await expect(copyButton).toBeVisible({ timeout: 10000 });
    await copyButton.click();

    // Verify button text changes
    await expect(page.getByRole('button', { name: /Copied!/i })).toBeVisible({ timeout: 2000 });
  });

  test('should reset form', async ({ page }) => {
    // Change form values
    await page.getByLabel(/Input Data Size \(GB\)/i).fill('100');
    await page.selectOption('select[name="jobType"]', 'sql');
    await page.getByLabel(/Application Name/i).fill('test-job');

    // Click reset
    await page.getByRole('button', { name: 'Reset' }).click();

    // Verify form is reset
    await expect(page.getByLabel(/Input Data Size \(GB\)/i)).toHaveValue('10');
    await expect(page.getByLabel(/Application Name/i)).toHaveValue('');
  });

  test('should handle API errors', async ({ page }) => {
    // Mock API error
    await page.route('**/api/v1/recommend', async (route) => {
      await route.abort('failed');
    });

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Should show error message
    await expect(page.getByText('Failed to get recommendation. Please try again.')).toBeVisible({ timeout: 10000 });
  });

  test('should show loading state during submission', async ({ page }) => {
    // Mock slow API response
    await page.route('**/api/v1/recommend', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          configuration: {
            executor_cores: 4,
            executor_memory_mb: 8192,
            num_executors: 10,
            driver_memory_mb: 4096
          },
          confidence: 0.85,
          method: 'hybrid'
        })
      });
    });

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Should show loading state
    await expect(page.getByText('Analyzing historical data and generating recommendations...')).toBeVisible({ timeout: 2000 });
  });
});
