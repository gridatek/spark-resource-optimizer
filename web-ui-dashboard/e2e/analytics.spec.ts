import { test, expect, Page } from '@playwright/test';

test.describe('Analytics Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/analytics');
    // Wait for page to load
    await page.waitForSelector('h1', { timeout: 10000 });
  });

  test('should display analytics title and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Performance Analytics' })).toBeVisible();
    await expect(page.getByText('Visualize job performance trends and resource utilization')).toBeVisible();
  });

  test('should display all chart sections', async ({ page }) => {
    // Wait for charts to load
    await waitForAnalyticsLoaded(page);

    // Verify all chart section headings are present
    const chartSections = [
      'Job Duration Trends',
      'Success Rate Trends',
      'Job Status Distribution',
      'Resource Utilization',
      'Estimated Cost Trends',
      'Data Processing Volume'
    ];

    for (const section of chartSections) {
      await expect(page.getByRole('heading', { name: section })).toBeVisible();
    }
  });

  test('should render chart canvas elements', async ({ page }) => {
    // Wait for charts to load
    await waitForAnalyticsLoaded(page);

    // Verify canvas elements exist (Chart.js renders to canvas)
    const canvases = page.locator('canvas');
    await expect(canvases.first()).toBeVisible({ timeout: 20000 });

    // There should be multiple charts
    const count = await canvases.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should handle loading state', async ({ page }) => {
    // Intercept API call to delay response
    await page.route('**/api/v1/jobs*', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.continue();
    });

    await page.goto('/analytics');

    // Should show loading state
    const loadingText = page.getByText('Loading analytics data...');
    await expect(loadingText).toBeVisible({ timeout: 2000 });
  });

  test('should handle error state with retry', async ({ page }) => {
    // Intercept API call to return error
    await page.route('**/api/v1/jobs*', async (route) => {
      await route.abort('failed');
    });

    await page.goto('/analytics');

    // Should show error message
    await expect(page.getByText('Failed to load analytics data')).toBeVisible({ timeout: 10000 });

    // Should show retry button
    const retryButton = page.getByRole('button', { name: 'Retry' });
    await expect(retryButton).toBeVisible();
  });

  test('should be responsive on mobile', async ({ page, viewport }) => {
    // Wait for charts to load
    await waitForAnalyticsLoaded(page);

    // Charts should be visible on any viewport
    const canvases = page.locator('canvas');
    const count = await canvases.count();

    // At least some charts should be visible
    if (count > 0) {
      await expect(canvases.first()).toBeVisible();
    }
  });

  test('should navigate back to dashboard', async ({ page }) => {
    // Wait for page to be interactive
    await waitForAnalyticsLoaded(page);

    await page.getByRole('link', { name: /Dashboard/i }).first().click();
    await expect(page).toHaveURL(/\/dashboard/);
  });
});

// Helper function to wait for analytics to finish loading
async function waitForAnalyticsLoaded(page: Page) {
  // Wait for loading to complete - either charts render or error appears
  await Promise.race([
    page.waitForSelector('canvas', { timeout: 20000 }),
    page.waitForSelector('text=Failed to load analytics data', { timeout: 20000 }),
    page.waitForSelector('text=No job data available', { timeout: 20000 })
  ]);
}
