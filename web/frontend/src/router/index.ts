import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';

import DashboardView from '@/views/DashboardView.vue';
import PipelineView from '@/views/PipelineView.vue';
import StreamsView from '@/views/StreamsView.vue';
import TelemetryView from '@/views/TelemetryView.vue';
import ConsoleView from '@/views/ConsoleView.vue';
import AdvisoriesView from '@/views/AdvisoriesView.vue';
import OrganismView from '@/views/OrganismView.vue';
import ModelLabView from '@/views/ModelLabView.vue';
import DataLabView from '@/views/DataLabView.vue';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'dashboard',
    component: DashboardView,
  },
  {
    path: '/pipeline',
    name: 'pipeline',
    component: PipelineView,
  },
  {
    path: '/streams',
    name: 'streams',
    component: StreamsView,
  },
  {
    path: '/telemetry',
    name: 'telemetry',
    component: TelemetryView,
  },
  {
    path: '/organism',
    name: 'organism',
    component: OrganismView,
  },
  {
    path: '/console',
    name: 'console',
    component: ConsoleView,
  },
  {
    path: '/advisories',
    name: 'advisories',
    component: AdvisoriesView,
  },
  {
    path: '/lab',
    name: 'lab',
    component: ModelLabView,
  },
  {
    path: '/datalab',
    name: 'datalab',
    component: DataLabView,
  },
];

export function createDashboardRouter(baseHref = '/') {
  return createRouter({
    history: createWebHistory(baseHref),
    routes,
  });
}
