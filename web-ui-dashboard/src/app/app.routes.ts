import { Routes } from '@angular/router';
import { Dashboard } from './components/dashboard/dashboard';
import { Recommendations } from './components/recommendations/recommendations';
import { ChartsComponent } from './components/charts/charts';
import { Monitoring } from './components/monitoring/monitoring';
import { Tuning } from './components/tuning/tuning';
import { Cost } from './components/cost/cost';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: Dashboard },
  { path: 'analytics', component: ChartsComponent },
  { path: 'monitoring', component: Monitoring },
  { path: 'tuning', component: Tuning },
  { path: 'cost', component: Cost },
  { path: 'recommendations', component: Recommendations },
  { path: '**', redirectTo: '/dashboard' }
];
