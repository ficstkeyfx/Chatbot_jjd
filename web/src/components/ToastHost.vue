<script setup>
import { Bell, CircleAlert, CircleCheck, Info, X } from '@lucide/vue'
import { dismissToast, toasts } from '../composables/useToast'

const icons = { error: CircleAlert, success: CircleCheck, reminder: Bell, info: Info }
</script>

<template>
  <div class="host" aria-live="polite" role="region" aria-label="Thông báo">
    <TransitionGroup name="toast">
      <div v-for="item in toasts" :key="item.id" class="toast" :class="item.type" role="status">
        <component :is="icons[item.type] || Info" :size="20" class="ico" />
        <div class="text">
          <strong v-if="item.title">{{ item.title }}</strong>
          <span>{{ item.message }}</span>
        </div>
        <button class="icon-btn small" aria-label="Đóng thông báo" @click="dismissToast(item.id)">
          <X :size="16" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.host {
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: 80;
  display: grid;
  gap: 10px;
  width: min(380px, calc(100vw - 32px));
  pointer-events: none;
}
.toast {
  pointer-events: auto;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 8px 12px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left-width: 4px;
  border-radius: 14px;
  box-shadow: var(--shadow);
}
.toast.error {
  border-left-color: var(--danger);
}
.toast.error .ico {
  color: var(--danger);
}
.toast.success {
  border-left-color: var(--primary);
}
.toast.success .ico {
  color: var(--primary);
}
.toast.reminder {
  border-left-color: var(--accent);
  background: var(--accent-soft);
}
.toast.reminder .ico {
  color: var(--accent);
}
.toast.info .ico {
  color: var(--text-soft);
}
.ico {
  flex: none;
  margin-top: 2px;
}
.text {
  flex: 1;
  display: grid;
  gap: 2px;
  font-size: 14px;
  min-width: 0;
  overflow-wrap: anywhere;
}
.small {
  width: 28px;
  height: 28px;
}
.toast-enter-active,
.toast-leave-active {
  transition: all 0.25s ease;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateX(24px);
}
@media (max-width: 560px) {
  .host {
    right: 16px;
    bottom: 96px;
  }
}
</style>
