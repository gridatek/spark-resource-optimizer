import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

export interface AppConfig {
  apiUrl: string;
}

@Injectable({
  providedIn: 'root'
})
export class ConfigService {
  private readonly http = inject(HttpClient);
  private config: AppConfig | null = null;

  async loadConfig(): Promise<void> {
    try {
      this.config = await firstValueFrom(
        this.http.get<AppConfig>('/config.json')
      );
    } catch {
      // Fallback to default config if config.json fails to load
      this.config = {
        apiUrl: 'http://localhost:8080'
      };
    }
  }

  get apiUrl(): string {
    return this.config?.apiUrl ?? 'http://localhost:8080';
  }

  getConfig(): AppConfig {
    if (!this.config) {
      throw new Error('Configuration not loaded. Ensure ConfigService.loadConfig() is called during app initialization.');
    }
    return this.config;
  }
}
