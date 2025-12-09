import { test, expect, Page } from '@playwright/test';

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    // Wait for either the dashboard content or loading state
    await page.waitForSelector('h1', { timeout: 10000 });
  });

  test('should display dashboard title and description', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Spark Resource Optimizer Dashboard' })).toBeVisible();
    await expect(page.getByText('Monitor and optimize your Spark job configurations')).toBeVisible();
  });

  test('should display statistics cards', async ({ page }) => {
    // Wait for loading to complete
    await waitForDashboardLoaded(page);

    // Check for statistics cards
    await expect(page.getByText('Total Jobs')).toBeVisible();
    await expect(page.getByText('Successful')).toBeVisible();
    await expect(page.getByText('Failed')).toBeVisible();
    await expect(page.getByText('Avg Duration')).toBeVisible();
  });

  test('should display auto-refresh toggle', async ({ page }) => {
    // Wait for dashboard to load
    await waitForDashboardLoaded(page);

    // Find the auto-refresh button (contains emoji and text)
    const autoRefreshButton = page.getByRole('button').filter({ hasText: /Auto-Refresh/i });
    await expect(autoRefreshButton).toBeVisible();

    // Check initial state (ON)
    await expect(autoRefreshButton).toContainText('ON');

    // Toggle off
    await autoRefreshButton.click();
    await expect(autoRefreshButton).toContainText('OFF');

    // Toggle back on
    await autoRefreshButton.click();
    await expect(autoRefreshButton).toContainText('ON');
  });

  test('should display last updated timestamp', async ({ page }) => {
    // Wait for data to load
    await waitForDashboardLoaded(page);
    await expect(page.getByText(/Last updated:/)).toBeVisible();
  });

  test('should display recent jobs table', async ({ page }) => {
    // Wait for dashboard to load
    await waitForDashboardLoaded(page);

    await expect(page.getByRole('heading', { name: 'Recent Jobs' })).toBeVisible();

    // Check for table headers - use locator within table to be more specific
    const table = page.locator('table');
    await expect(table.getByText('App ID')).toBeVisible();
    await expect(table.getByText('App Name')).toBeVisible();
    await expect(table.getByText('Status')).toBeVisible();
    await expect(table.getByText('Duration')).toBeVisible();
  });

  test('should handle loading state', async ({ page }) => {
    // Intercept API call to delay response
    await page.route('**/api/v1/jobs*', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.continue();
    });

    await page.goto('/dashboard');

    // Should show loading state - use a shorter timeout since we delayed the API
    const loadingText = page.getByText('Loading dashboard data...');
    await expect(loadingText).toBeVisible({ timeout: 2000 });
  });

  test('should be responsive on mobile', async ({ page, viewport }) => {
    if (!viewport || viewport.width >= 768) {
      test.skip();
      return;
    }

    // On mobile, the menu button should be visible
    const menuButton = page.getByRole('button', { name: /toggle menu/i });
    await expect(menuButton).toBeVisible();

    // Open mobile menu
    await menuButton.click();

    // Check mobile nav links are visible
    await expect(page.getByRole('link', { name: /Dashboard/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Analytics/i })).toBeVisible();
  });

  test('should navigate to analytics page', async ({ page }) => {
    // Wait for page to be interactive
    await waitForDashboardLoaded(page);

    // Click on Analytics link (use first to avoid multiple matches)
    await page.getByRole('link', { name: /Analytics/i }).first().click();
    await expect(page).toHaveURL(/\/analytics/);
    await expect(page.getByRole('heading', { name: /Performance Analytics/i })).toBeVisible();
  });

  test('should navigate to recommendations page', async ({ page }) => {
    // Wait for page to be interactive
    await waitForDashboardLoaded(page);

    // Click on Recommendations link
    await page.getByRole('link', { name: /Recommendations/i }).first().click();
    await expect(page).toHaveURL(/\/recommendations/);
    await expect(page.getByRole('heading', { name: /Get Resource Recommendations/i })).toBeVisible();
  });
});

// Helper function to wait for dashboard to finish loading
async function waitForDashboardLoaded(page: Page) {
  // Wait for loading to disappear or content to appear
  await Promise.race([
    page.waitForSelector('text=Recent Jobs', { timeout: 15000 }),
    page.waitForSelector('text=No jobs found', { timeout: 15000 }),
    page.waitForSelector('text=Failed to load', { timeout: 15000 })
  ]);
}
