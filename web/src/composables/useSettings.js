import { reactive } from 'vue'
import { api } from './api'
import { toast } from './useToast'

export const ROLES = ['llm', 'vision', 'stt', 'tts']

const empty = () => ({
  loaded: false,
  profile: {
    llm: { provider: 'local', model: '', base_url: '' },
    vision: { provider: 'ocr', model: '', base_url: '' },
    stt: { provider: 'off', model: '', base_url: '' },
    tts: { provider: 'off', model: '', base_url: '', voice: '' },
  },
  effective: { llm: '', vision: '', stt: '', tts: '' },
  keys: {}, // nhà cung cấp -> đã lưu key của bạn
  server_keys: {}, // nhà cung cấp -> máy chủ có key dùng chung
  providers: [],
  special: {},
  transcript: false,
  tts_speed: 0.9,
  tts_speed_range: [0.5, 1.3],
  privacy_notice: '',
})

export const settings = reactive(empty())

function apply(data) {
  Object.assign(settings, data, { loaded: true })
}

export function resetSettings() {
  Object.assign(settings, empty())
}

export async function loadSettings() {
  apply(await api.getSettings())
}

/** Lưu một phần cài đặt; trả về true nếu thành công, đã hiển thị thông báo lỗi nếu thất bại. */
export async function saveSettings(patch, { quiet = false } = {}) {
  try {
    apply(await api.saveSettings(patch))
    if (!quiet) toast('Đã lưu cài đặt', { type: 'success', duration: 2000 })
    return true
  } catch (error) {
    toast(error.message, { type: 'error' })
    return false
  }
}

export function applySettings(data) {
  apply(data)
}

export const providerById = (id) => settings.providers.find((item) => item.id === id)

/** Giọng nói được bật khi cả hai vai trò nghe/đọc không tắt. */
export const sttEnabled = () => settings.profile.stt.provider !== 'off'
export const ttsEnabled = () => settings.profile.tts.provider !== 'off'
