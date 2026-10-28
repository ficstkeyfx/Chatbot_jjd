<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { X } from '@lucide/vue'

defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, default: '' },
  wide: { type: Boolean, default: false },
})
const emit = defineEmits(['close'])
const panel = ref(null)

function onKey(event) {
  if (event.key === 'Escape') emit('close')
}

onMounted(async () => {
  document.addEventListener('keydown', onKey)
  await nextTick()
  panel.value?.querySelector('input, textarea, button:not(.close)')?.focus()
})
onBeforeUnmount(() => document.removeEventListener('keydown', onKey))
</script>

<template>
  <Teleport to="body">
    <div class="backdrop" @mousedown.self="emit('close')">
      <section
        ref="panel"
        class="panel"
        :class="{ wide }"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
      >
        <header class="head">
          <div>
            <h2>{{ title }}</h2>
            <p v-if="subtitle">{{ subtitle }}</p>
          </div>
          <button class="icon-btn close" aria-label="Đóng" @click="emit('close')"><X :size="20" /></button>
        </header>
        <div class="body"><slot /></div>
        <footer v-if="$slots.footer" class="foot"><slot name="footer" /></footer>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: grid;
  place-items: center;
  padding: 16px;
  background: rgba(10, 25, 22, 0.5);
  backdrop-filter: blur(3px);
  animation: fade 0.15s ease-out;
}
.panel {
  width: min(560px, 100%);
  max-height: min(760px, 100%);
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow-lg);
  animation: rise 0.2s ease-out;
}
.panel.wide {
  width: min(680px, 100%);
}
.head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 20px 20px 12px 24px;
}
h2 {
  margin: 0;
  font-size: 19px;
}
.head p {
  margin: 2px 0 0;
  color: var(--text-soft);
  font-size: 13.5px;
}
.body {
  padding: 4px 24px 20px;
  overflow-y: auto;
}
.foot {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 14px 24px;
  border-top: 1px solid var(--border);
}
@keyframes fade {
  from {
    opacity: 0;
  }
}
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(12px) scale(0.98);
  }
}
@media (max-width: 560px) {
  .backdrop {
    place-items: end center;
    padding: 0;
  }
  .panel {
    border-radius: 22px 22px 0 0;
    max-height: 92%;
  }
}
</style>
