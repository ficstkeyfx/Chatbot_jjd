import { computed, reactive, ref } from 'vue'
import { api, sessionId } from './api'
import { toast } from './useToast'

// Đồng hồ đếm ngược chạy ở trình duyệt, đồng bộ lại với máy chủ mỗi khi có sự kiện.
const now = ref(Date.now())
setInterval(() => (now.value = Date.now()), 1000)

const raw = reactive({ pomodoro: null, eye: null, syncedAt: Date.now() })
const busy = ref(false)
let reminderListener = null
let syncListener = null

export function onSync(listener) {
  syncListener = listener
}

export function onReminder(listener) {
  reminderListener = listener
}

function apply(state) {
  raw.pomodoro = state.pomodoro
  raw.eye = state.eye
  raw.syncedAt = Date.now()
}

function remaining(entry) {
  if (!entry) return 0
  return Math.max(0, entry.remaining_seconds - (now.value - raw.syncedAt) / 1000)
}

export const pomodoro = computed(() => {
  const entry = raw.pomodoro
  if (!entry) return null
  const left = remaining(entry)
  return {
    ...entry,
    left,
    progress: entry.phase_seconds ? Math.min(1, Math.max(0, 1 - left / entry.phase_seconds)) : 0,
  }
})

export const eye = computed(() => {
  const entry = raw.eye
  if (!entry) return null
  const left = remaining(entry)
  return {
    ...entry,
    left,
    progress: Math.min(1, Math.max(0, 1 - left / entry.interval_seconds)),
  }
})

export const timersBusy = busy

export async function refreshTimers() {
  try {
    apply(await api.timers())
  } catch {
    /* lần đồng bộ sau sẽ thử lại */
  }
}

async function run(action) {
  busy.value = true
  try {
    apply(await action())
    return true
  } catch (error) {
    toast(error.message, { type: 'error' })
    return false
  } finally {
    busy.value = false
  }
}

export const startPomodoro = (work, rest, cycles) => run(() => api.startPomodoro(work, rest, cycles))
export const setEyeReminder = (enabled) => run(() => api.setEye(enabled))
export const stopTimer = (target) => run(() => api.stopTimer(target))

export function applyTimerState(state) {
  apply(state)
}

export function resetTimers() {
  apply({ pomodoro: null, eye: null })
}

// --- Thông báo nhắc nhở: SSE + thông báo của trình duyệt + âm báo ngắn ---

function chime() {
  try {
    const Context = window.AudioContext || window.webkitAudioContext
    if (!Context) return
    const audio = new Context()
    const tone = (frequency, start) => {
      const oscillator = audio.createOscillator()
      const gain = audio.createGain()
      oscillator.frequency.value = frequency
      gain.gain.setValueAtTime(0.0001, audio.currentTime + start)
      gain.gain.exponentialRampToValueAtTime(0.18, audio.currentTime + start + 0.03)
      gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + start + 0.45)
      oscillator.connect(gain).connect(audio.destination)
      oscillator.start(audio.currentTime + start)
      oscillator.stop(audio.currentTime + start + 0.5)
    }
    tone(660, 0)
    tone(880, 0.22)
    setTimeout(() => audio.close(), 1200)
  } catch {
    /* âm thanh là tính năng phụ */
  }
}

const REMINDER_TITLES = {
  pomodoro_break: 'Đến giờ nghỉ',
  pomodoro_focus: 'Quay lại tập trung',
  pomodoro_done: 'Hoàn thành Pomodoro',
  eye: 'Nghỉ mắt 20-20-20',
}

function handleReminder(event) {
  const title = REMINDER_TITLES[event.kind] || 'Nhắc nhở'
  toast(event.text, { title, type: 'reminder', duration: 9000 })
  chime()
  if ('Notification' in window && Notification.permission === 'granted' && document.hidden) {
    try {
      new Notification(title, { body: event.text, icon: '/favicon.svg' })
    } catch {
      /* một số trình duyệt di động không cho tạo Notification trực tiếp */
    }
  }
  reminderListener?.(event, title)
}

export function requestNotificationPermission() {
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission().catch(() => {})
  }
}

let source = null

export function connectEvents() {
  if (source) return
  source = new EventSource(`/api/events?sid=${encodeURIComponent(sessionId)}`)
  source.onmessage = async (message) => {
    let event
    try {
      event = JSON.parse(message.data)
    } catch {
      return
    }
    if (event.type === 'reminder') handleReminder(event)
    if (event.type === 'sync' || (event.type === 'conversation' && event.source === 'zalo')) {
      await syncListener?.()
    }
    // Với mọi sự kiện, đồng bộ lại trạng thái để phase/chu kỳ luôn đúng.
    await refreshTimers()
  }
  // EventSource tự kết nối lại; sau khi nối lại thì đồng bộ ngay.
  source.onopen = () => {
    refreshTimers()
    syncListener?.()
  }
}

export function disconnectEvents() {
  source?.close()
  source = null
}
