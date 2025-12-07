import { Routes } from '@angular/router';
import { Dashboard } from './components/dashboard/dashboard.component';
import { Recommendations } from './components/recommendations/recommendations.component';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: Dashboard },
  { path: 'recommendations', component: Recommendations },
  { path: '**', redirectTo: '/dashboard' }
];
