<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { Bot } from '@lucide/vue'
import EmptyState from './EmptyState.vue'
import MessageBubble from './MessageBubble.vue'
import { messages, sending } from '../composables/useChat'

const scroller = ref(null)

async function toBottom(smooth = true) {
  await nextTick()
  const el = scroller.value
  if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
}

watch(() => [messages.length, sending.value], () => toBottom())
onMounted(() => toBottom(false))
</script>

<template>
  <div ref="scroller" class="scroller">
    <div class="inner" role="log" aria-live="polite" aria-label="Cuộc trò chuyện">
      <EmptyState v-if="!messages.length && !sending" />
      <MessageBubble v-for="message in messages" :key="message.id" :message="message" />
      <div v-if="sending" class="typing" aria-label="Trợ lý đang trả lời">
        <span class="avatar"><Bot :size="18" /></span>
        <span class="dots"><i /><i /><i /></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.scroller {
  flex: 1;
  overflow-y: auto;
  padding: 20px 20px 8px;
}
.inner {
  width: min(820px, 100%);
  min-height: 100%;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.typing {
  display: flex;
  align-items: center;
  gap: 10px;
}
.avatar {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 12px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.dots {
  display: inline-flex;
  gap: 5px;
  padding: 14px 16px;
  border-radius: 18px;
  border-top-left-radius: 6px;
  background: var(--surface);
  border: 1px solid var(--border);
}
.dots i {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--muted);
  animation: bounce 1.2s infinite ease-in-out;
}
.dots i:nth-child(2) {
  animation-delay: 0.15s;
}
.dots i:nth-child(3) {
  animation-delay: 0.3s;
}
@keyframes bounce {
  0%,
  60%,
  100% {
    transform: translateY(0);
    opacity: 0.5;
  }
  30% {
    transform: translateY(-5px);
    opacity: 1;
  }
}
@media (max-width: 560px) {
  .scroller {
    padding: 14px 12px 6px;
  }
}
</style>
