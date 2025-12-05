import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { RecommendationsComponent } from './components/recommendations/recommendations.component';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'recommendations', component: RecommendationsComponent },
  { path: '**', redirectTo: '/dashboard' }
];
