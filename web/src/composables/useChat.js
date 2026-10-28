import { reactive, ref } from 'vue'
import { api, sessionId } from './api'
import { auth } from './useAuth'
import { loadSettings, ttsEnabled } from './useSettings'
import { applyTimerState, refreshTimers } from './useTimers'
import { toast } from './useToast'
import { openDialog, requestPick } from './useUi'

let counter = 0
const newId = () => `${Date.now().toString(36)}-${(counter++).toString(36)}`

// Lịch sử được máy chủ lưu trong cơ sở dữ liệu; trình duyệt chỉ giữ bản đang hiển thị.
export const messages = reactive([])
export const sending = ref(false)
export const persistent = ref(false)
export const playingId = ref(null)
export const speakingId = ref(null) // tin đang được tải giọng đọc

export async function loadHistory() {
  const owner = auth.user
  const data = await api.messages()
  if (owner === auth.user) applyHistory(data)
}

function applyHistory(data) {
  persistent.value = data.persistent
  messages.splice(
    0,
    messages.length,
    ...data.messages.map((item) => ({
      id: `db-${item.id}`,
      ts: item.ts,
      role: item.role,
      kind: item.kind === 'reminder' ? 'text' : item.kind,
      imageUrl: item.image_url ? `${item.image_url}?sid=${encodeURIComponent(sessionId)}` : undefined,
      text: item.kind === 'voice' ? '' : item.text,
      transcript: item.kind === 'voice' ? item.text : undefined,
      alert: item.alert,
    })),
  )
}

let remoteSyncPending = false
export async function syncRemoteHistory() {
  if (sending.value) {
    remoteSyncPending = true
    return
  }
  const owner = auth.user
  const data = await api.messages()
  if (owner !== auth.user) return
  if (sending.value) remoteSyncPending = true
  else applyHistory(data)
}

function push(message) {
  const entry = { id: newId(), ts: Date.now(), kind: 'text', ...message }
  messages.push(entry)
  return messages[messages.length - 1]
}

export function addSystemMessage(text, extra = {}) {
  return push({ role: 'system', text, ...extra })
}

// --- phát giọng đọc ---

let audio = null

function stopAudio() {
  if (audio) {
    audio.pause()
    URL.revokeObjectURL(audio.src)
    audio = null
  }
  playingId.value = null
}

export async function speak(id, text) {
  if (playingId.value === id) {
    stopAudio()
    return
  }
  stopAudio()
  speakingId.value = id
  try {
    const blob = await api.tts(text)
    const player = new Audio(URL.createObjectURL(blob))
    audio = player
    playingId.value = id
    player.onended = () => {
      if (audio === player) stopAudio()
    }
    player.onerror = () => {
      toast('Không phát được âm thanh.', { type: 'error' })
      if (audio === player) stopAudio()
    }
    await player.play()
  } catch (error) {
    stopAudio()
    toast(error.message || 'Không phát được âm thanh.', { type: 'error' })
  } finally {
    speakingId.value = null
  }
}

export { stopAudio }

// --- xử lý kết quả từ máy chủ ---

async function applyAction(action) {
  if (!action) return
  if (action.type === 'timers') await refreshTimers()
  else if (action.type === 'settings') await loadSettings().catch(() => {})
  else if (action.type === 'clear') messages.splice(0, messages.length)
  else if (action.type === 'open') {
    if (action.target === 'emotion') requestPick(true)
    else openDialog(action.target)
  }
  else if (action.type === 'speak') speak(`speak-${newId()}`, action.text)
}

function pushBot(result) {
  const entry = push({ role: 'bot', text: result.reply, alert: result.alert || null })
  return entry
}

function pushError(error) {
  if (error.name === 'AbortError') return
  push({ role: 'bot', text: error.message || 'Đã xảy ra lỗi.', error: true })
}

let controller = null

export function cancelPending() {
  controller?.abort()
}

async function perform(task) {
  if (sending.value) return false
  sending.value = true
  controller = new AbortController()
  try {
    return await task(controller.signal)
  } finally {
    sending.value = false
    controller = null
    if (remoteSyncPending) {
      remoteSyncPending = false
      await syncRemoteHistory().catch(() => {})
    }
  }
}

export async function sendText(text) {
  const value = text.trim()
  if (!value) return
  await perform(async (signal) => {
    push({ role: 'user', text: value })
    try {
      const result = await api.chat(value, signal)
      pushBot(result)
      await applyAction(result.action)
    } catch (error) {
      pushError(error)
    }
  })
}

export async function sendImage(file, caption, expression) {
  await perform(async (signal) => {
    const text = caption.trim()
    const entry = push({
      role: 'user',
      kind: 'image',
      text,
      imageUrl: URL.createObjectURL(file),
    })
    try {
      const result = await api.image(file, text, expression, signal, 'camera')
      if (result.image_url) {
        URL.revokeObjectURL(entry.imageUrl)
        entry.imageUrl = `${result.image_url}?sid=${encodeURIComponent(sessionId)}`
      }
      pushBot(result)
    } catch (error) {
      pushError(error)
    }
  })
}

export async function sendVoice(blob) {
  await perform(async (signal) => {
    const extension = blob.type.includes('mp4') ? 'm4a' : blob.type.includes('ogg') ? 'ogg' : 'webm'
    const placeholder = push({ role: 'user', kind: 'voice', text: '', pending: true })
    try {
      const result = await api.voice(blob, `voice.${extension}`, signal)
      placeholder.pending = false
      placeholder.transcript = result.transcript
      const reply = pushBot(result)
      if (ttsEnabled() && !result.alert) speak(reply.id, reply.text)
    } catch (error) {
      placeholder.pending = false
      placeholder.text = 'Voice note (chưa xử lý được)'
      pushError(error)
    }
  })
}

export async function summarize(payload, signal) {
  const result = await api.summarize(payload, signal)
  return result.reply
}

export async function deleteEverything() {
  stopAudio()
  try {
    await api.deleteMe()
  } catch (error) {
    toast(error.message, { type: 'error' })
    return false
  }
  messages.splice(0, messages.length)
  // Đang đăng nhập: tài khoản đã bị xóa cùng dữ liệu, quay về màn hình đăng nhập.
  if (auth.required) {
    auth.user = null
    return true
  }
  await loadSettings().catch(() => {})
  await refreshTimers()
  return true
}

/** Dọn trạng thái hội thoại khi đăng xuất để tài khoản sau không thấy dữ liệu của tài khoản trước. */
export function resetChat() {
  remoteSyncPending = false
  stopAudio()
  messages.splice(0, messages.length)
}

export async function newConversation() {
  stopAudio()
  try {
    await api.clearHistory()
  } catch (error) {
    toast(error.message, { type: 'error' })
    return
  }
  messages.splice(0, messages.length)
  applyTimerState((await api.timers().catch(() => ({ pomodoro: null, eye: null }))))
}
