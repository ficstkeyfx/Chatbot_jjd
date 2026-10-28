<script setup>
import { computed } from 'vue'
import {
  AudioLines,
  Bell,
  Bot,
  Check,
  Copy,
  Image as ImageIcon,
  Loader,
  Phone,
  Square,
  TriangleAlert,
  Volume2,
} from '@lucide/vue'
import { formatClock, formatText } from '../composables/formatText'
import { playingId, speak, speakingId } from '../composables/useChat'
import { settings } from '../composables/useSettings'
import { toast } from '../composables/useToast'
import { ref } from 'vue'

const props = defineProps({ message: { type: Object, required: true } })

const shownText = computed(() => {
  const message = props.message
  if (message.kind === 'voice') return settings.transcript ? message.transcript || '' : ''
  // Hide the generated caption in saved image messages as well as new uploads.
  if (message.role === 'user' && message.kind === 'image' &&
      message.text === 'Phân tích biểu cảm trong ảnh') return ''
  return message.text
})
const blocks = computed(() => formatText(shownText.value))
const isBot = computed(() => props.message.role === 'bot')
const isSystem = computed(() => props.message.role === 'system')
const copied = ref(false)

async function copy() {
  try {
    await navigator.clipboard.writeText(props.message.text)
    copied.value = true
    setTimeout(() => (copied.value = false), 1500)
  } catch {
    toast('Không sao chép được nội dung.', { type: 'error' })
  }
}
</script>

<template>
  <!-- Nhắc nhở hẹn giờ / thông báo hệ thống -->
  <div v-if="isSystem" class="system">
    <Bell :size="15" />
    <span>{{ message.text }}</span>
    <time>{{ formatClock(message.ts) }}</time>
  </div>

  <article v-else class="row" :class="{ bot: isBot, user: !isBot }">
    <span v-if="isBot" class="avatar" aria-hidden="true"><Bot :size="18" /></span>

    <div class="stack">
      <!-- Cảnh báo khẩn cấp: nổi bật hơn nội dung thường -->
      <div v-if="message.alert" class="alert" :class="message.alert" role="alert">
        <TriangleAlert :size="20" />
        <div>
          <strong>{{ message.alert === 'crisis' ? 'Bạn có thể đang gặp nguy hiểm' : 'Cần được khám mắt khẩn' }}</strong>
          <p v-for="(block, index) in blocks" :key="index">
            <template v-for="(part, i) in block.parts || []" :key="i">{{ part.text }}</template>
          </p>
          <a class="call" href="tel:115"><Phone :size="16" /> Gọi cấp cứu 115</a>
        </div>
      </div>

      <div v-else class="bubble" :class="{ error: message.error }">
        <img v-if="message.imageUrl" class="photo" :src="message.imageUrl" alt="Ảnh bạn đã gửi" />
        <p v-else-if="message.kind === 'image'" class="chip"><ImageIcon :size="15" /> Ảnh đã gửi</p>

        <p v-if="message.kind === 'voice'" class="chip voice">
          <Loader v-if="message.pending" :size="15" class="spin" />
          <AudioLines v-else :size="15" />
          {{ message.pending ? 'Đang nghe…' : 'Voice note' }}
        </p>

        <template v-for="(block, index) in blocks" :key="index">
          <p v-if="block.type === 'p'">
            <template v-for="(part, i) in block.parts" :key="i">
              <strong v-if="part.bold">{{ part.text }}</strong>
              <template v-else>{{ part.text }}</template>
            </template>
          </p>
          <component :is="block.type" v-else>
            <li v-for="(item, n) in block.items" :key="n">
              <template v-for="(part, i) in item" :key="i">
                <strong v-if="part.bold">{{ part.text }}</strong>
                <template v-else>{{ part.text }}</template>
              </template>
            </li>
          </component>
        </template>
      </div>

      <div class="meta">
        <time>{{ formatClock(message.ts) }}</time>
        <template v-if="isBot && !message.error">
          <button
            class="mini"
            :aria-label="playingId === message.id ? 'Dừng đọc' : 'Nghe câu trả lời'"
            :disabled="speakingId === message.id"
            @click="speak(message.id, message.text)"
          >
            <Loader v-if="speakingId === message.id" :size="14" class="spin" />
            <Square v-else-if="playingId === message.id" :size="14" />
            <Volume2 v-else :size="14" />
          </button>
          <button class="mini" aria-label="Sao chép" @click="copy">
            <Check v-if="copied" :size="14" />
            <Copy v-else :size="14" />
          </button>
        </template>
      </div>
    </div>
  </article>
</template>

<style scoped>
.row {
  display: flex;
  gap: 10px;
  max-width: 100%;
  animation: pop 0.22s ease-out;
}
.row.user {
  justify-content: flex-end;
}
.avatar {
  flex: none;
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  margin-top: 2px;
  border-radius: 12px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.stack {
  display: grid;
  gap: 4px;
  min-width: 0;
  max-width: min(640px, 86%);
}
.user .stack {
  justify-items: end;
}
.bubble {
  padding: 11px 15px;
  border-radius: 18px;
  overflow-wrap: anywhere;
  box-shadow: var(--shadow-sm);
}
.bot .bubble {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top-left-radius: 6px;
}
.user .bubble {
  background: var(--primary);
  color: var(--on-primary);
  border-top-right-radius: 6px;
}
.bubble.error {
  background: var(--danger-soft);
  border-color: transparent;
  color: var(--danger);
}
.bubble p {
  margin: 0;
}
.bubble p + p,
.bubble p + ul,
.bubble p + ol,
.bubble ul + p,
.bubble ol + p {
  margin-top: 8px;
}
.bubble ul,
.bubble ol {
  margin: 0;
  padding-left: 20px;
}
.bubble li + li {
  margin-top: 4px;
}
.bubble li::marker {
  color: var(--primary);
}
.photo {
  display: block;
  max-width: min(280px, 100%);
  max-height: 260px;
  margin-bottom: 8px;
  border-radius: 12px;
  object-fit: cover;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
  font-size: 13px;
  opacity: 0.9;
}
.meta {
  display: flex;
  align-items: center;
  gap: 2px;
  font-size: 11.5px;
  color: var(--muted);
  padding: 0 4px;
}
.meta time {
  margin-right: 6px;
}
.mini {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 8px;
  color: var(--muted);
}
.mini:hover:not(:disabled) {
  background: var(--bg-soft);
  color: var(--text);
}
.alert {
  display: flex;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid;
}
.alert.crisis {
  background: var(--danger-soft);
  border-color: var(--danger);
  color: var(--text);
}
.alert.crisis > svg {
  color: var(--danger);
}
.alert.eye {
  background: var(--accent-soft);
  border-color: var(--accent);
}
.alert.eye > svg {
  color: var(--accent);
}
.alert > svg {
  flex: none;
  margin-top: 2px;
}
.alert strong {
  display: block;
  margin-bottom: 4px;
}
.alert p {
  margin: 4px 0;
}
.call {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 9px 16px;
  border-radius: 12px;
  background: var(--danger);
  color: #fff;
  font-weight: 700;
  text-decoration: none;
}
.system {
  align-self: center;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: 92%;
  padding: 8px 14px;
  border-radius: 99px;
  background: var(--accent-soft);
  color: var(--text);
  font-size: 13.5px;
  animation: pop 0.22s ease-out;
}
.system svg {
  flex: none;
  color: var(--accent);
}
.system time {
  flex: none;
  color: var(--muted);
  font-size: 11.5px;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@keyframes pop {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
}
</style>
