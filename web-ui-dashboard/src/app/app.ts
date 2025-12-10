import { Component, signal, ChangeDetectionStrategy } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="flex flex-col min-h-screen bg-gray-50">
      <!-- Navigation Header -->
      <header class="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-8 py-4 flex justify-between items-center">
          <div class="flex items-center">
            <h1 class="text-2xl font-bold text-gray-900">⚡ {{ title() }}</h1>
          </div>

          <!-- Desktop Navigation -->
          <nav class="hidden md:flex gap-2">
            <a
              routerLink="/dashboard"
              routerLinkActive="bg-blue-500 text-white"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900"
            >
              <span class="text-lg">📊</span>
              Dashboard
            </a>
            <a
              routerLink="/monitoring"
              routerLinkActive="bg-blue-500 text-white"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900"
            >
              <span class="text-lg">📡</span>
              Monitoring
            </a>
            <a
              routerLink="/tuning"
              routerLinkActive="bg-blue-500 text-white"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900"
            >
              <span class="text-lg">🔧</span>
              Tuning
            </a>
            <a
              routerLink="/cost"
              routerLinkActive="bg-blue-500 text-white"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900"
            >
              <span class="text-lg">💰</span>
              Cost
            </a>
            <a
              routerLink="/analytics"
              routerLinkActive="bg-blue-500 text-white"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900"
            >
              <span class="text-lg">📈</span>
              Analytics
            </a>
            <a
              routerLink="/recommendations"
              routerLinkActive="bg-blue-500 text-white"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900"
            >
              <span class="text-lg">🎯</span>
              Recommendations
            </a>
          </nav>

          <!-- Mobile Menu Button -->
          <button
            (click)="toggleMenu()"
            class="md:hidden bg-transparent border-none text-2xl cursor-pointer p-2 text-gray-700"
            aria-label="Toggle menu"
          >
            <span class="block w-6 h-6 text-center">{{ isMenuOpen() ? '✕' : '☰' }}</span>
          </button>
        </div>

        <!-- Mobile Navigation -->
        @if (isMenuOpen()) {
          <nav class="md:hidden flex flex-col px-8 py-4 border-t border-gray-200">
            <a
              routerLink="/dashboard"
              routerLinkActive="bg-blue-500 text-white"
              (click)="toggleMenu()"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900 w-full justify-start"
            >
              <span class="text-lg">📊</span>
              Dashboard
            </a>
            <a
              routerLink="/monitoring"
              routerLinkActive="bg-blue-500 text-white"
              (click)="toggleMenu()"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900 w-full justify-start"
            >
              <span class="text-lg">📡</span>
              Monitoring
            </a>
            <a
              routerLink="/tuning"
              routerLinkActive="bg-blue-500 text-white"
              (click)="toggleMenu()"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900 w-full justify-start"
            >
              <span class="text-lg">🔧</span>
              Tuning
            </a>
            <a
              routerLink="/cost"
              routerLinkActive="bg-blue-500 text-white"
              (click)="toggleMenu()"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900 w-full justify-start"
            >
              <span class="text-lg">💰</span>
              Cost
            </a>
            <a
              routerLink="/analytics"
              routerLinkActive="bg-blue-500 text-white"
              (click)="toggleMenu()"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900 w-full justify-start"
            >
              <span class="text-lg">📈</span>
              Analytics
            </a>
            <a
              routerLink="/recommendations"
              routerLinkActive="bg-blue-500 text-white"
              (click)="toggleMenu()"
              class="flex items-center gap-2 px-6 py-3 rounded-md no-underline text-gray-700 font-medium transition-all hover:bg-gray-100 hover:text-gray-900 w-full justify-start"
            >
              <span class="text-lg">🎯</span>
              Recommendations
            </a>
          </nav>
        }
      </header>

      <!-- Main Content -->
      <main class="flex-1 w-full">
        <router-outlet />
      </main>

      <!-- Footer -->
      <footer class="bg-white border-t border-gray-200 mt-auto">
        <div class="max-w-7xl mx-auto px-8 py-8 flex flex-col md:flex-row justify-between items-center gap-4 md:gap-0">
          <p class="text-gray-600 m-0">&copy; 2025 Gridatek - Spark Resource Optimizer</p>
          <div class="flex flex-col md:flex-row gap-2 md:gap-6">
            <a
              href="https://github.com/gridatek/spark-resource-optimizer"
              target="_blank"
              rel="noopener"
              class="text-gray-700 no-underline text-sm transition-colors hover:text-blue-500"
            >
              GitHub
            </a>
            <a
              href="https://github.com/gridatek/spark-resource-optimizer/docs"
              target="_blank"
              rel="noopener"
              class="text-gray-700 no-underline text-sm transition-colors hover:text-blue-500"
            >
              Documentation
            </a>
          </div>
        </div>
      </footer>
    </div>
  `
})
export class App {
  protected readonly title = signal('Spark Resource Optimizer');
  protected readonly isMenuOpen = signal(false);

  toggleMenu() {
    this.isMenuOpen.update(value => !value);
  }
}
