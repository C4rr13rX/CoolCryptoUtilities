import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';

import DashboardView from '@/views/DashboardView.vue';
import PipelineView from '@/views/PipelineView.vue';
import StreamsView from '@/views/StreamsView.vue';
import TelemetryView from '@/views/TelemetryView.vue';
import WalletView from '@/views/WalletView.vue';
import AdvisoriesView from '@/views/AdvisoriesView.vue';
import OrganismView from '@/views/OrganismView.vue';
import ModelLabView from '@/views/ModelLabView.vue';
import DataLabView from '@/views/DataLabView.vue';
import GuardianView from '@/views/GuardianView.vue';
import CodeGraphView from '@/views/CodeGraphView.vue';
import SecureSettingsView from '@/views/SecureSettingsView.vue';
import ApiIntegrationsView from '@/views/ApiIntegrationsView.vue';
import BrandDozerView from '@/views/BrandDozerView.vue';
import U53RxRobotView from '@/views/U53RxRobotView.vue';
import AddressBookView from '@/views/AddressBookView.vue';
import C0d3rView from '@/views/C0d3rView.vue';
import AudioLabView from '@/views/AudioLabView.vue';

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
    path: '/wallet',
    name: 'wallet',
    component: WalletView,
  },
  {
    path: '/integrations',
    name: 'integrations',
    component: ApiIntegrationsView,
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
  {
    path: '/guardian',
    name: 'guardian',
    component: GuardianView,
  },
  {
    path: '/codegraph',
    name: 'codegraph',
    component: CodeGraphView,
  },
  {
    path: '/settings',
    name: 'settings',
    component: SecureSettingsView,
  },
  {
    path: '/addressbook',
    name: 'addressbook',
    component: AddressBookView,
    meta: { title: 'Address Book' },
  },
  {
    path: '/c0d3r',
    name: 'c0d3r',
    component: C0d3rView,
    meta: { title: 'c0d3r' },
  },
  {
    path: '/branddozer',
    name: 'branddozer',
    component: BrandDozerView,
    meta: { title: 'BrandDozer' },
  },
  {
    path: '/branddozer/solo',
    name: 'branddozer_solo',
    component: BrandDozerView,
    meta: { layout: 'solo', title: 'BrandDozer' },
  },
  {
    path: '/u53rxr080t',
    name: 'u53rxr080t',
    component: U53RxRobotView,
  },
  {
    path: '/audiolab',
    name: 'audiolab',
    component: AudioLabView,
    meta: { title: 'Audio Lab' },
  },
];

export function createDashboardRouter(baseHref = '/') {
  return createRouter({
    history: createWebHistory(baseHref),
    routes,
  });
}
