import { test, expect } from '@playwright/test';

test.describe('Analytics Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/analytics');
  });

  test('should display analytics title and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Performance Analytics' })).toBeVisible();
    await expect(page.getByText('Visualize job performance trends and resource utilization')).toBeVisible();
  });

  test('should display job duration trends chart', async ({ page }) => {
    // Wait for charts to load
    await page.waitForSelector('canvas', { timeout: 15000 });

    await expect(page.getByRole('heading', { name: 'Job Duration Trends' })).toBeVisible();

    // Check for canvas elements (charts)
    const canvases = await page.locator('canvas').all();
    expect(canvases.length).toBeGreaterThan(0);
  });

  test('should display success rate trends chart', async ({ page }) => {
    await page.waitForSelector('canvas', { timeout: 15000 });
    await expect(page.getByRole('heading', { name: 'Success Rate Trends' })).toBeVisible();
  });

  test('should display job status distribution chart', async ({ page }) => {
    await page.waitForSelector('canvas', { timeout: 15000 });
    await expect(page.getByRole('heading', { name: 'Job Status Distribution' })).toBeVisible();
  });

  test('should display resource utilization chart', async ({ page }) => {
    await page.waitForSelector('canvas', { timeout: 15000 });
    await expect(page.getByRole('heading', { name: 'Resource Utilization' })).toBeVisible();
  });

  test('should display cost trends chart', async ({ page }) => {
    await page.waitForSelector('canvas', { timeout: 15000 });
    await expect(page.getByRole('heading', { name: 'Estimated Cost Trends' })).toBeVisible();
    await expect(page.getByText('* Cost estimates based on standard compute pricing')).toBeVisible();
  });

  test('should display data processing volume chart', async ({ page }) => {
    await page.waitForSelector('canvas', { timeout: 15000 });
    await expect(page.getByRole('heading', { name: 'Data Processing Volume' })).toBeVisible();
  });

  test('should handle loading state', async ({ page }) => {
    // Intercept API call to delay response
    await page.route('**/api/v1/jobs*', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 100));
      await route.continue();
    });

    await page.goto('/analytics');

    // Should show loading spinner initially
    await expect(page.getByText('Loading analytics data...')).toBeVisible({ timeout: 1000 });
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

  test('should display all chart sections', async ({ page }) => {
    // Wait for page to load
    await page.waitForSelector('canvas', { timeout: 15000 });

    // Verify all major sections are present
    const sections = [
      'Job Duration Trends',
      'Success Rate Trends',
      'Job Status Distribution',
      'Resource Utilization',
      'Estimated Cost Trends',
      'Data Processing Volume'
    ];

    for (const section of sections) {
      await expect(page.getByRole('heading', { name: section })).toBeVisible();
    }
  });

  test('should be responsive on mobile', async ({ page, viewport }) => {
    await page.waitForSelector('canvas', { timeout: 15000 });

    // Charts should be visible and responsive
    const canvases = await page.locator('canvas').all();
    for (const canvas of canvases) {
      await expect(canvas).toBeVisible();
    }
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: /dashboard/i }).first().click();
    await expect(page).toHaveURL(/\/dashboard/);
  });
});
