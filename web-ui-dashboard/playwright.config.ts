import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Spark Resource Optimizer E2E tests.
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 4 : undefined,
  reporter: process.env.CI ? 'github' : 'html',

  // Global timeout settings
  timeout: 60000, // 60 seconds per test
  expect: {
    timeout: 10000, // 10 seconds for expect assertions
  },

  use: {
    baseURL: 'http://localhost:4200',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: process.env.CI ? 'on-first-retry' : 'off',

    // Action timeouts
    actionTimeout: 15000,
    navigationTimeout: 30000,
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    // Mobile viewports
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
  ],

  webServer: [
    {
      command: 'pnpm start',
      url: 'http://localhost:4200',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
  ],
});
