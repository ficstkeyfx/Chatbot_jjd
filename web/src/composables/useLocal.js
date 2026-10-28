import { reactive } from 'vue'
import { api } from './api'

// Tài nguyên có sẵn trên máy chủ (model Ollama, giọng Piper), chỉ để gợi ý khi chọn "Chạy trên máy chủ".
export const local = reactive({
  loaded: false,
  loading: false,
  llm_models: [],
  llm_available: false,
  voices: { default: null, installed: [], downloadable: [] },
})

export async function loadLocal(force = false) {
  if (local.loading || (local.loaded && !force)) return
  local.loading = true
  try {
    const data = await api.localResources()
    local.llm_models = data.llm_models || []
    local.llm_available = !!data.llm_available
    local.voices = data.tts_voices || local.voices
    local.loaded = true
  } catch {
    /* chỉ là gợi ý; lỗi thì để trống */
  } finally {
    local.loading = false
  }
}
