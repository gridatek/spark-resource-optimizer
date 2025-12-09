import { test, expect, Page } from '@playwright/test';

test.describe('Recommendations Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/recommendations');
    // Wait for page to load
    await page.waitForSelector('h1', { timeout: 10000 });
  });

  test('should display recommendations title and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Get Resource Recommendations' })).toBeVisible();
    await expect(page.getByText('Get optimal Spark configuration recommendations for your job')).toBeVisible();
  });

  test('should display recommendation form', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Job Requirements' })).toBeVisible();

    // Check form fields using label text (partial match to handle asterisk)
    await expect(page.getByLabel(/Input Data Size/i)).toBeVisible();
    await expect(page.getByLabel(/Job Type/i)).toBeVisible();
    await expect(page.getByLabel(/Application Name/i)).toBeVisible();
    await expect(page.getByLabel(/Recommendation Method/i)).toBeVisible();

    // Check buttons
    await expect(page.getByRole('button', { name: /Get Recommendation/i })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reset' })).toBeVisible();
  });

  test('should have default form values', async ({ page }) => {
    const inputSizeField = page.getByLabel(/Input Data Size/i);
    await expect(inputSizeField).toHaveValue('10');
  });

  test('should validate required fields', async ({ page }) => {
    // Clear the input size field
    const inputSizeField = page.getByLabel(/Input Data Size/i);
    await inputSizeField.clear();

    // Submit button should be disabled when required field is empty
    const submitButton = page.getByRole('button', { name: /Get Recommendation/i });
    await expect(submitButton).toBeDisabled();
  });

  test('should submit recommendation request', async ({ page }) => {
    // Fill in the form
    await page.getByLabel(/Input Data Size/i).fill('50');
    await page.locator('select#jobType').selectOption('ml');
    await page.getByLabel(/Application Name/i).fill('test-ml-job');

    // Submit form
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await expect(page.getByRole('heading', { name: 'Recommended Configuration' })).toBeVisible({ timeout: 20000 });

    // Verify configuration cards are displayed
    await expect(page.getByText('Executor Cores')).toBeVisible();
    await expect(page.getByText('Executor Memory')).toBeVisible();
  });

  test('should display recommendation results', async ({ page }) => {
    // Submit with default values
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await waitForRecommendationResult(page);

    // Check confidence badge
    await expect(page.getByText('Confidence:')).toBeVisible();

    // Check configuration cards
    await expect(page.getByText('Executor Cores')).toBeVisible();
    await expect(page.getByText('Executor Memory')).toBeVisible();
    await expect(page.getByText('Number of Executors')).toBeVisible();
    await expect(page.getByText('Driver Memory')).toBeVisible();

    // Check predicted metrics (if present)
    const predictedMetrics = page.getByText('Predicted Performance');
    if (await predictedMetrics.isVisible().catch(() => false)) {
      await expect(page.getByText('Duration:')).toBeVisible();
      await expect(page.getByText('Estimated Cost:')).toBeVisible();
    }

    // Check spark command section
    await expect(page.getByRole('heading', { name: 'Spark Submit Command' })).toBeVisible();
    await expect(page.getByText(/spark-submit/)).toBeVisible();
  });

  test('should display resource allocation chart', async ({ page }) => {
    // Submit with default values
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await waitForRecommendationResult(page);

    // Check for resource allocation chart
    await expect(page.getByRole('heading', { name: 'Resource Allocation' })).toBeVisible();

    // Wait for chart canvas to render
    const chartCanvas = page.locator('canvas').first();
    await expect(chartCanvas).toBeVisible({ timeout: 20000 });
  });

  test('should display similar jobs visualization when available', async ({ page }) => {
    // Submit with default values
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await waitForRecommendationResult(page);

    // Similar jobs section is optional - check if it appears
    const similarJobsHeading = page.getByRole('heading', { name: 'Similar Jobs Analysis' });
    const hasSimilarJobs = await similarJobsHeading.isVisible({ timeout: 5000 }).catch(() => false);

    if (hasSimilarJobs) {
      await expect(page.getByText('Similarity:')).toBeVisible();
    }
  });

  test('should display cost-performance indicator when available', async ({ page }) => {
    // Submit with default values
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await waitForRecommendationResult(page);

    // Cost-performance indicator is optional - check if it appears
    const costPerfHeading = page.getByRole('heading', { name: 'Cost-Performance Indicator' });
    const hasCostPerf = await costPerfHeading.isVisible({ timeout: 5000 }).catch(() => false);

    if (hasCostPerf) {
      await expect(page.getByText('Cost Efficient')).toBeVisible();
      await expect(page.getByText('High Performance')).toBeVisible();
    }
  });

  test('should copy spark command to clipboard', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    // Submit with default values
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Wait for results
    await waitForRecommendationResult(page);

    // Find and click copy button (has emoji prefix)
    const copyButton = page.getByRole('button').filter({ hasText: /Copy/i });
    await expect(copyButton).toBeVisible({ timeout: 15000 });
    await copyButton.click();

    // Verify button text changes to indicate success
    await expect(page.getByRole('button').filter({ hasText: /Copied/i })).toBeVisible({ timeout: 3000 });
  });

  test('should reset form', async ({ page }) => {
    // Change form values
    await page.getByLabel(/Input Data Size/i).fill('100');
    await page.locator('select#jobType').selectOption('sql');
    await page.getByLabel(/Application Name/i).fill('test-job');

    // Click reset
    await page.getByRole('button', { name: 'Reset' }).click();

    // Verify form is reset to defaults
    await expect(page.getByLabel(/Input Data Size/i)).toHaveValue('10');
    await expect(page.getByLabel(/Application Name/i)).toHaveValue('');
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Submit with zero input size (should trigger validation or error)
    await page.getByLabel(/Input Data Size/i).fill('0');

    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Should show error message
    const errorMessage = page.locator('.bg-red-100, [class*="error"], [class*="text-red"]');
    await expect(errorMessage.first()).toBeVisible({ timeout: 10000 });
  });

  test('should show loading state during submission', async ({ page }) => {
    // Intercept API to slow it down
    await page.route('**/api/v1/recommend*', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.continue();
    });

    // Click submit button
    await page.getByRole('button', { name: /Get Recommendation/i }).click();

    // Should show loading indicator
    const loadingIndicator = page.getByText('Analyzing historical data and generating recommendations...');
    await expect(loadingIndicator).toBeVisible({ timeout: 2000 });
  });
});

// Helper function to wait for recommendation result
async function waitForRecommendationResult(page: Page) {
  await Promise.race([
    page.waitForSelector('text=Recommended Configuration', { timeout: 20000 }),
    page.waitForSelector('text=Failed to get recommendation', { timeout: 20000 }),
    page.waitForSelector('.bg-red-100', { timeout: 20000 })
  ]);
}
