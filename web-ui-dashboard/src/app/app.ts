import { Component, signal } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class App {
  protected readonly title = signal('Spark Resource Optimizer');
  protected readonly isMenuOpen = signal(false);

  toggleMenu() {
    this.isMenuOpen.update(value => !value);
  }
}
