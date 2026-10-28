<script setup>
import { computed, ref } from 'vue'
import { Brain, Loader, Mic2, UserRound, Volume2 } from '@lucide/vue'
import AccountPanel from './AccountPanel.vue'
import ModalShell from './ModalShell.vue'
import ModelRole from './ModelRole.vue'
import { api } from '../composables/api'
import { auth } from '../composables/useAuth'
import { speak } from '../composables/useChat'
import { saveSettings, settings, ttsEnabled } from '../composables/useSettings'
import { toast } from '../composables/useToast'
import { closeDialog } from '../composables/useUi'

const tab = ref('models') // models | voice | account
const tabs = computed(() => [
  { id: 'models', label: 'Mô hình', icon: Brain },
  { id: 'voice', label: 'Giọng nói', icon: Mic2 },
  { id: 'account', label: auth.required ? 'Tài khoản' : 'Dữ liệu', icon: UserRound },
])

const voiceInfo = ref(null)
const checking = ref(false)
const testing = ref(false)

const setTranscript = (event) => saveSettings({ transcript: event.target.checked }, { quiet: true })

let speedTimer = null
function onSpeed(event) {
  const value = Number(event.target.value)
  settings.tts_speed = value
  clearTimeout(speedTimer)
  speedTimer = setTimeout(() => saveSettings({ tts_speed: value }, { quiet: true }), 250)
}

const speedLabel = computed(() => {
  const value = settings.tts_speed
  if (value < 0.65) return 'Rất chậm'
  if (value < 0.85) return 'Chậm'
  if (value <= 1.05) return 'Vừa phải'
  return 'Nhanh'
})

async function testVoice() {
  testing.value = true
  clearTimeout(speedTimer)
  if (await saveSettings({ tts_speed: settings.tts_speed }, { quiet: true })) {
    await speak('settings-test', 'Xin chào, mình là trợ lý sức khỏe. Bạn hãy hít vào chậm, rồi thở ra thật nhẹ nhàng nhé.')
  }
  testing.value = false
}

async function checkVoice() {
  checking.value = true
  try {
    voiceInfo.value = await api.voiceStatus()
  } catch (error) {
    toast(error.message, { type: 'error' })
  } finally {
    checking.value = false
  }
}
</script>

<template>
  <ModalShell
    title="Cài đặt"
    :subtitle="auth.user ? `Tài khoản ${auth.user.username}: mọi lựa chọn và API key chỉ áp dụng cho bạn.` : 'Mọi thay đổi chỉ áp dụng cho trình duyệt này.'"
    wide
    @close="closeDialog"
  >
    <nav class="tabs" role="tablist" aria-label="Nhóm cài đặt">
      <button
        v-for="item in tabs"
        :key="item.id"
        role="tab"
        :aria-selected="tab === item.id"
        :class="{ on: tab === item.id }"
        @click="tab = item.id"
      >
        <component :is="item.icon" :size="16" /> {{ item.label }}
      </button>
    </nav>

    <div v-show="tab === 'models'" role="tabpanel">
      <ModelRole
        role="llm"
        title="Mô hình trả lời"
        hint="Mô hình trò chuyện và tóm tắt. Chọn nhà cung cấp rồi nhập API key của bạn."
      />
      <ModelRole
        role="vision"
        title="Đọc ảnh (VLM / OCR)"
        hint="Hiểu ảnh khuôn mặt, đơn thuốc, xét nghiệm. Chọn 'Chỉ đọc chữ (OCR)' nếu không muốn gửi ảnh cho mô hình."
      />
    </div>

    <div v-show="tab === 'voice'" role="tabpanel">
      <ModelRole role="stt" title="Nhận dạng giọng nói" hint="Chuyển lời bạn nói thành chữ khi bấm micro." />
      <ModelRole role="tts" title="Đọc thành tiếng" hint="Giọng đọc câu trả lời của trợ lý." />

      <section class="extra">
        <label class="speed">
          <span class="between">Tốc độ đọc <b>{{ settings.tts_speed.toFixed(2) }}× · {{ speedLabel }}</b></span>
          <input
            type="range"
            class="range"
            :min="settings.tts_speed_range[0]"
            :max="settings.tts_speed_range[1]"
            step="0.05"
            :value="settings.tts_speed"
            :disabled="!ttsEnabled()"
            @input="onSpeed"
          />
        </label>
        <label class="check">
          <input type="checkbox" :checked="settings.transcript" @change="setTranscript" />
          Hiển thị nội dung tôi nói trong khung chat
        </label>
        <div class="row">
          <button class="btn btn-ghost" :disabled="checking" @click="checkVoice">
            <Loader v-if="checking" :size="15" class="spin" /> Kiểm tra tình trạng
          </button>
          <button class="btn btn-soft" :disabled="testing || !ttsEnabled()" @click="testVoice">
            <Loader v-if="testing" :size="15" class="spin" /> <Volume2 v-else :size="15" /> Nghe thử
          </button>
        </div>
        <ul v-if="voiceInfo" class="status">
          <li v-for="role in ['stt', 'tts']" :key="role" :class="{ ok: voiceInfo[role].ready }">
            <b>{{ role === 'stt' ? 'Nhận dạng' : 'Đọc to' }}:</b> {{ voiceInfo[role].detail }}
          </li>
        </ul>
        <p class="hint">{{ settings.privacy_notice }}</p>
      </section>
    </div>

    <div v-show="tab === 'account'" role="tabpanel"><AccountPanel /></div>
  </ModalShell>
</template>

<style scoped>
.tabs {
  display: flex;
  gap: 4px;
  padding: 4px;
  margin-bottom: 14px;
  background: var(--bg-soft);
  border-radius: 12px;
}
.tabs button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 8px;
  border-radius: 9px;
  font-weight: 600;
  color: var(--text-soft);
}
.tabs button.on {
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-sm);
}
.extra {
  display: grid;
  gap: 12px;
  padding: 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 16px;
}
.speed {
  display: grid;
  gap: 6px;
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text-soft);
}
.between {
  display: flex;
  justify-content: space-between;
}
.between b {
  color: var(--primary-strong);
}
.range {
  width: 100%;
  accent-color: var(--primary);
}
.check {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
}
.check input {
  width: 18px;
  height: 18px;
  accent-color: var(--primary);
}
.row {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  flex-wrap: wrap;
}
.status {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 6px;
}
.status li {
  padding: 8px 12px;
  border-radius: 10px;
  background: var(--danger-soft);
  color: var(--danger);
  font-size: 13px;
}
.status li.ok {
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.hint {
  margin: 0;
  font-size: 12.5px;
  color: var(--text-soft);
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
