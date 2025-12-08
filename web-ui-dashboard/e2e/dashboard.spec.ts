import { test, expect } from '@playwright/test';

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display dashboard title and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Spark Resource Optimizer Dashboard' })).toBeVisible();
    await expect(page.getByText('Monitor and optimize your Spark job configurations')).toBeVisible();
  });

  test('should display statistics cards', async ({ page }) => {
    // Wait for data to load
    await page.waitForSelector('[class*="grid-cols"]', { timeout: 10000 });

    // Check for statistics cards
    await expect(page.getByText('Total Jobs')).toBeVisible();
    await expect(page.getByText('Successful')).toBeVisible();
    await expect(page.getByText('Failed')).toBeVisible();
    await expect(page.getByText('Avg Duration')).toBeVisible();
  });

  test('should display auto-refresh toggle', async ({ page }) => {
    const autoRefreshButton = page.getByRole('button', { name: /auto-refresh/i });
    await expect(autoRefreshButton).toBeVisible();

    // Check initial state
    await expect(autoRefreshButton).toContainText('Auto-Refresh ON');

    // Toggle off
    await autoRefreshButton.click();
    await expect(autoRefreshButton).toContainText('Auto-Refresh OFF');

    // Toggle back on
    await autoRefreshButton.click();
    await expect(autoRefreshButton).toContainText('Auto-Refresh ON');
  });

  test('should display last updated timestamp', async ({ page }) => {
    // Wait for initial data load
    await page.waitForSelector('text=Last updated:', { timeout: 10000 });
    await expect(page.getByText(/Last updated:/)).toBeVisible();
  });

  test('should display recent jobs table', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Recent Jobs' })).toBeVisible();

    // Check for table headers
    await expect(page.getByText('App ID')).toBeVisible();
    await expect(page.getByText('App Name')).toBeVisible();
    await expect(page.getByText('Status')).toBeVisible();
    await expect(page.getByText('Duration')).toBeVisible();
  });

  test('should handle loading state', async ({ page }) => {
    // Intercept API call to delay response
    await page.route('**/api/v1/jobs*', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 100));
      await route.continue();
    });

    await page.goto('/dashboard');

    // Should show loading spinner initially
    await expect(page.getByText('Loading dashboard data...')).toBeVisible({ timeout: 1000 });
  });

  test('should be responsive on mobile', async ({ page, viewport }) => {
    // Test mobile menu
    if (viewport && viewport.width < 768) {
      const menuButton = page.getByRole('button', { name: /toggle menu/i });
      await expect(menuButton).toBeVisible();

      // Open mobile menu
      await menuButton.click();
      await expect(page.getByRole('link', { name: /dashboard/i })).toBeVisible();
    }
  });

  test('should navigate to analytics page', async ({ page }) => {
    await page.getByRole('link', { name: /analytics/i }).first().click();
    await expect(page).toHaveURL(/\/analytics/);
    await expect(page.getByRole('heading', { name: /Performance Analytics/i })).toBeVisible();
  });

  test('should navigate to recommendations page', async ({ page }) => {
    await page.getByRole('link', { name: /recommendations/i }).first().click();
    await expect(page).toHaveURL(/\/recommendations/);
    await expect(page.getByRole('heading', { name: /Get Resource Recommendations/i })).toBeVisible();
  });
});
