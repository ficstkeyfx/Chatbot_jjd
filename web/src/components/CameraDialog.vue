<script setup>
import { nextTick, onBeforeUnmount, ref } from 'vue'
import { Camera, ImagePlus, RefreshCw } from '@lucide/vue'
import ModalShell from './ModalShell.vue'
import { createCameraSession } from '../composables/cameraSession'
import { cameraErrorMessage } from '../composables/cameraErrors'

const emit = defineEmits(['close', 'capture', 'upload'])
const video = ref(null)
const preview = ref('')
const error = ref('')
const loading = ref(false)
const ready = ref(false)
const capturing = ref(false)
const attempted = ref(false)
const diagnostic = ref('')
const facing = ref('user')
const session = createCameraSession()
let photo = null
let disposed = false

function clearPhoto() {
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  photo = null
}

async function start() {
  clearPhoto()
  ready.value = false
  error.value = ''
  diagnostic.value = ''
  attempted.value = true
  loading.value = true
  try {
    if (!window.isSecureContext) throw new DOMException('Insecure context', 'SecurityError')
    const policy = document.permissionsPolicy || document.featurePolicy
    if (policy?.allowsFeature && !policy.allowsFeature('camera')) {
      throw new DOMException('Camera blocked by Permissions Policy', 'SecurityError')
    }
    const stream = await session.open(facing.value)
    if (!stream || disposed) return
    await nextTick()
    if (disposed || !video.value) return
    video.value.srcObject = stream
    await video.value.play()
  } catch (cause) {
    if (disposed) return
    session.stop()
    let permission = 'unknown'
    try { permission = (await navigator.permissions.query({ name: 'camera' })).state } catch { /* not supported by all browsers */ }
    if (disposed) return
    const policy = document.permissionsPolicy || document.featurePolicy
    const policyAllowed = !policy?.allowsFeature || policy.allowsFeature('camera')
    error.value = cameraErrorMessage(cause, { secure: window.isSecureContext, policyAllowed, permission })
    diagnostic.value = `${cause?.name || 'CameraUnavailable'}: ${String(cause?.message || '').slice(0, 220)} · quyền: ${permission} · secure: ${window.isSecureContext}`
  } finally {
    if (!disposed) loading.value = false
  }
}

async function capture() {
  const source = video.value
  if (!ready.value || capturing.value || !source?.videoWidth) return
  capturing.value = true
  try {
    const scale = Math.min(1, 1600 / Math.max(source.videoWidth, source.videoHeight))
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(source.videoWidth * scale)
    canvas.height = Math.round(source.videoHeight * scale)
    canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9))
    if (disposed) return
    if (!blob) throw new Error('Không tạo được ảnh, hãy thử chụp lại.')
    photo = new File([blob], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' })
    preview.value = URL.createObjectURL(photo)
    ready.value = false
    session.stop()
  } catch (cause) {
    error.value = cause.message || 'Không chụp được ảnh, hãy thử lại.'
  } finally {
    capturing.value = false
  }
}

function usePhoto() {
  if (photo) emit('capture', photo)
}
function switchCamera() {
  facing.value = facing.value === 'user' ? 'environment' : 'user'
  start()
}
onBeforeUnmount(() => {
  disposed = true
  session.stop()
  clearPhoto()
})
</script>

<template>
  <ModalShell title="Chụp ảnh từ camera" subtitle="Chụp rõ khuôn mặt, đủ sáng. Ảnh chỉ được gửi khi bạn bấm Gửi trong khung chat." @close="emit('close')">
    <div class="viewfinder">
      <img v-if="preview" :src="preview" alt="Ảnh vừa chụp" />
      <video v-else-if="attempted" ref="video" autoplay muted playsinline :class="{ mirrored: facing === 'user' }" aria-label="Hình ảnh camera trực tiếp" @loadeddata="ready = true" />
      <p v-if="!attempted">Bấm Bật camera, rồi chọn Cho phép khi trình duyệt hỏi.</p>
      <p v-if="loading" role="status">Đang chờ camera. Hãy kiểm tra yêu cầu cấp quyền cạnh thanh địa chỉ.</p>
    </div>
    <p v-if="error" class="camera-error" role="alert">{{ error }}</p>
    <small v-if="diagnostic" class="diagnostic">{{ diagnostic }}</small>
    <div class="actions">
      <template v-if="preview">
        <button class="btn btn-ghost" @click="start"><RefreshCw :size="17" /> Chụp lại</button>
        <button class="btn btn-primary" @click="usePhoto">Dùng ảnh này</button>
      </template>
      <template v-else>
        <button class="btn btn-ghost" :disabled="!ready || loading || capturing" @click="switchCamera"><RefreshCw :size="17" /> Đổi camera</button>
        <button v-if="error || !attempted" class="btn btn-primary" :disabled="loading" @click="start">{{ attempted ? 'Thử lại' : 'Bật camera' }}</button>
        <button v-else class="btn btn-primary" :disabled="!ready || loading || capturing" @click="capture"><Camera :size="17" /> Chụp ảnh</button>
      </template>
      <button class="btn btn-ghost" @click="emit('upload')"><ImagePlus :size="17" /> Chọn ảnh có sẵn</button>
    </div>
  </ModalShell>
</template>

<style scoped>
.viewfinder { background: #10201d; border-radius: 14px; overflow: hidden; min-height: 180px; position: relative; }
.viewfinder video, .viewfinder img { display: block; width: 100%; max-height: 48vh; object-fit: contain; }
.mirrored { transform: scaleX(-1); }
.viewfinder p { color: white; text-align: center; padding: 12px; }
.actions { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 16px; }
.camera-error { color: var(--danger); font-size: 14px; margin-top: 12px; }
.diagnostic { display: block; overflow-wrap: anywhere; color: var(--text-soft); margin-top: 8px; }
</style>
