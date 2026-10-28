<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Camera, Check, ImagePlus, Mic, Send, Smile, Square, X } from '@lucide/vue'
import CameraDialog from './CameraDialog.vue'
import { cancelPending, sendImage, sendText, sendVoice, sending } from '../composables/useChat'
import { useRecorder } from '../composables/useRecorder'
import { formatMmSs } from '../composables/formatText'
import { settings, sttEnabled } from '../composables/useSettings'
import { toast } from '../composables/useToast'
import { ui } from '../composables/useUi'

const text = ref('')
const input = ref(null)
const fileInput = ref(null)
const cameraOpen = ref(false)
const image = ref(null) // { file, url, expression }
const dragging = ref(false)
const recorder = useRecorder()

const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const canSend = computed(() => !sending.value && (text.value.trim() || image.value))
const voiceOff = computed(() => settings.loaded && !sttEnabled())

function grow() {
  const el = input.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 160) + 'px'
}
watch(text, () => nextTick(grow))

function attach(file, expression = false) {
  if (!file) return
  if (!file.type.startsWith('image/')) {
    toast('Chỉ hỗ trợ tệp ảnh. Với PDF/DOCX hãy dùng công cụ Tóm tắt bài học.', { type: 'error' })
    return
  }
  if (file.size > MAX_IMAGE_BYTES) {
    toast('Ảnh vượt quá 10 MB.', { type: 'error' })
    return
  }
  clearImage()
  image.value = { file, url: URL.createObjectURL(file), expression }
  input.value?.focus()
}

function clearImage() {
  if (image.value) URL.revokeObjectURL(image.value.url)
  image.value = null
  if (fileInput.value) fileInput.value.value = ''
}

function onPick(event) {
  attach(event.target.files?.[0], pickExpression)
}

let pickExpression = false
function openPicker(expression = false) {
  pickExpression = expression
  fileInput.value?.click()
}
function attachCameraPhoto(file) {
  cameraOpen.value = false
  attach(file, true)
}
function uploadFromCamera() {
  cameraOpen.value = false
  openPicker(true)
}
watch(
  () => ui.pick.n,
  () => openPicker(ui.pick.expression),
)

async function submit() {
  if (!canSend.value) return
  const message = text.value
  const attached = image.value
  text.value = ''
  image.value = null
  if (fileInput.value) fileInput.value.value = ''
  await nextTick(grow)
  if (attached) {
    await sendImage(attached.file, message, attached.expression)
    URL.revokeObjectURL(attached.url)
  } else {
    await sendText(message)
  }
  input.value?.focus()
}

function onKeydown(event) {
  // Không gửi khi đang gõ dở bằng bộ gõ tiếng Việt (Telex/VNI).
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    submit()
  }
}

function onPaste(event) {
  const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith('image/'))
  if (file) {
    event.preventDefault()
    attach(file)
  }
}

function onDrop(event) {
  dragging.value = false
  attach(event.dataTransfer?.files?.[0])
}

async function startRecording() {
  if (voiceOff.value) {
    toast('Nhận dạng giọng nói đang tắt. Hãy bật trong Cài đặt → Giọng nói.', { type: 'info' })
    return
  }
  try {
    await recorder.start()
  } catch (error) {
    const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError'
    toast(
      denied
        ? 'Chưa được cấp quyền micro. Hãy cho phép micro trong trình duyệt (cần HTTPS hoặc localhost).'
        : 'Không mở được micro trên thiết bị này.',
      { type: 'error' },
    )
  }
}

async function finishRecording() {
  const blob = await recorder.stop()
  if (blob && blob.size > 1000) await sendVoice(blob)
  else toast('Đoạn ghi âm quá ngắn.', { type: 'info', duration: 2500 })
}

onMounted(() => input.value?.focus())
onBeforeUnmount(clearImage)
</script>

<template>
  <div
    class="composer-wrap"
    :class="{ dragging }"
    @dragover.prevent="dragging = true"
    @dragleave.prevent="dragging = false"
    @drop.prevent="onDrop"
  >
    <div class="composer">
      <div v-if="image" class="preview">
        <img :src="image.url" alt="Ảnh sẽ gửi" />
        <div class="info">
          <strong>{{ image.file.name || 'Ảnh' }}</strong>
          <button
            class="toggle"
            :class="{ on: image.expression }"
            type="button"
            :aria-pressed="image.expression"
            @click="image.expression = !image.expression"
          >
            <Smile :size="14" /> Nhận gợi ý từ biểu cảm
          </button>
        </div>
        <button class="icon-btn" aria-label="Bỏ ảnh" @click="clearImage"><X :size="18" /></button>
      </div>

      <!-- Đang ghi âm -->
      <div v-if="recorder.recording.value" class="recording" role="status">
        <i class="rec" />
        <span class="time">{{ formatMmSs(recorder.seconds.value) }}</span>
        <span class="wave" aria-hidden="true"><b v-for="n in 22" :key="n" :style="{ '--i': n }" /></span>
        <button class="icon-btn" aria-label="Hủy ghi âm" @click="recorder.cancel()"><X :size="20" /></button>
        <button class="send" aria-label="Gửi ghi âm" @click="finishRecording"><Check :size="20" /></button>
      </div>

      <div v-else class="line">
        <button class="icon-btn" aria-label="Đính kèm ảnh" title="Đính kèm ảnh" @click="openPicker(false)">
          <ImagePlus :size="21" />
        </button>
        <button class="icon-btn" aria-label="Chụp ảnh từ camera" title="Chụp ảnh từ camera" :disabled="sending" @click="cameraOpen = true">
          <Camera :size="21" />
        </button>
        <textarea
          ref="input"
          v-model="text"
          rows="1"
          maxlength="4000"
          placeholder="Nhắn cho trợ lý…"
          title="Enter để gửi, Shift+Enter để xuống dòng"
          aria-label="Nội dung tin nhắn"
          @keydown="onKeydown"
          @paste="onPaste"
        />
        <button v-if="sending" class="send stop" aria-label="Dừng" @click="cancelPending">
          <Square :size="17" fill="currentColor" />
        </button>
        <button v-else-if="canSend" class="send" aria-label="Gửi" @click="submit"><Send :size="19" /></button>
        <button
          v-else
          class="send mic"
          :disabled="!recorder.supported.value"
          :title="recorder.supported.value ? 'Ghi âm giọng nói' : 'Trình duyệt không hỗ trợ ghi âm'"
          aria-label="Ghi âm giọng nói"
          @click="startRecording"
        >
          <Mic :size="20" />
        </button>
      </div>
      <input ref="fileInput" type="file" accept="image/*" hidden @change="onPick" />
    </div>
    <p class="disclaimer">
      Chỉ hỗ trợ ban đầu, không thay thế bác sĩ. Khẩn cấp hãy gọi <a href="tel:115">115</a>.
    </p>
    <div v-if="dragging" class="drop">Thả ảnh vào đây để gửi</div>
  </div>
  <CameraDialog v-if="cameraOpen" @close="cameraOpen = false" @capture="attachCameraPhoto" @upload="uploadFromCamera" />
</template>

<style scoped>
.composer-wrap {
  position: relative;
  padding: 8px 20px calc(10px + env(safe-area-inset-bottom));
}
.composer {
  width: min(820px, 100%);
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 8px;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.composer:focus-within {
  border-color: var(--primary);
  box-shadow: var(--shadow), 0 0 0 3px var(--primary-soft);
}
.line {
  display: flex;
  align-items: flex-end;
  gap: 6px;
}
textarea {
  flex: 1;
  min-width: 0;
  resize: none;
  border: 0;
  outline: none;
  background: transparent;
  padding: 9px 6px;
  max-height: 160px;
  line-height: 1.5;
}
textarea::placeholder {
  color: var(--muted);
}
.send {
  flex: none;
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 14px;
  background: var(--primary);
  color: var(--on-primary);
  transition: background 0.15s, transform 0.1s;
}
.send:hover:not(:disabled) {
  background: var(--primary-strong);
}
.send:active:not(:disabled) {
  transform: scale(0.95);
}
.send.mic {
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.send.stop {
  background: var(--danger-soft);
  color: var(--danger);
}
.line .icon-btn {
  width: 42px;
  height: 42px;
}
.preview {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 6px 10px;
}
.preview img {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  object-fit: cover;
  border: 1px solid var(--border);
}
.info {
  flex: 1;
  min-width: 0;
  display: grid;
  gap: 6px;
  justify-items: start;
}
.info strong {
  font-size: 13.5px;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 11px;
  border-radius: 99px;
  border: 1px solid var(--border);
  font-size: 12.5px;
  color: var(--text-soft);
}
.toggle.on {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--text);
}
.recording {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 2px 4px 2px 12px;
  min-height: 46px;
}
.rec {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--danger);
  animation: blink 1.1s infinite;
}
.time {
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.wave {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 3px;
  height: 28px;
  overflow: hidden;
}
.wave b {
  width: 3px;
  height: 100%;
  border-radius: 3px;
  background: var(--primary);
  animation: wave 1s ease-in-out infinite;
  animation-delay: calc(var(--i) * -0.07s);
  transform-origin: center;
}
.disclaimer {
  width: min(820px, 100%);
  margin: 8px auto 0;
  text-align: center;
  font-size: 12px;
  color: var(--muted);
}
.disclaimer a {
  color: var(--danger);
  font-weight: 700;
}
.drop {
  position: absolute;
  inset: 4px 12px;
  display: grid;
  place-items: center;
  border: 2px dashed var(--primary);
  border-radius: 22px;
  background: color-mix(in srgb, var(--primary-soft) 88%, transparent);
  color: var(--primary-strong);
  font-weight: 700;
  pointer-events: none;
}
@keyframes blink {
  50% {
    opacity: 0.25;
  }
}
@keyframes wave {
  0%,
  100% {
    transform: scaleY(0.2);
  }
  50% {
    transform: scaleY(1);
  }
}
@media (max-width: 560px) {
  .composer-wrap {
    padding: 6px 10px calc(8px + env(safe-area-inset-bottom));
  }
  .disclaimer {
    font-size: 11px;
  }
}
</style>
