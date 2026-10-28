// Lớp gọi API: gắn mã phiên vào mọi yêu cầu và chuẩn hoá lỗi tiếng Việt.

const SID_KEY = 'hb.sid'

function makeSessionId() {
  const raw =
    globalThis.crypto && crypto.randomUUID
      ? crypto.randomUUID()
      : Array.from({ length: 4 }, () => Math.random().toString(36).slice(2, 10)).join('')
  return raw.replace(/[^A-Za-z0-9_-]/g, '').padEnd(16, '0').slice(0, 64)
}

function loadSessionId() {
  try {
    let sid = localStorage.getItem(SID_KEY)
    if (!sid || !/^[A-Za-z0-9_-]{16,64}$/.test(sid)) {
      sid = makeSessionId()
      localStorage.setItem(SID_KEY, sid)
    }
    return sid
  } catch {
    return makeSessionId()
  }
}

export const sessionId = loadSessionId()

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.status = status
  }
}

// Được đặt bởi useAuth: gọi khi máy chủ trả 401 (phiên đăng nhập hết hạn hoặc chưa đăng nhập).
let unauthorizedHandler = null
export function onUnauthorized(handler) {
  unauthorizedHandler = handler
}

async function request(path, { method = 'GET', json, form, signal, raw = false, auth = false } = {}) {
  // X-HB-Client chống CSRF: trình duyệt không tự thêm header này vào yêu cầu từ trang khác.
  const headers = { 'X-Session-Id': sessionId, 'X-HB-Client': '1' }
  let body
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(json)
  } else if (form) {
    body = form
  }
  let response
  try {
    response = await fetch(path, { method, headers, body, signal })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new ApiError('Không kết nối được máy chủ. Hãy kiểm tra server đang chạy.')
  }
  if (!response.ok) {
    let message = `Lỗi ${response.status}`
    try {
      const data = await response.json()
      message = data.error || (Array.isArray(data.detail) ? 'Dữ liệu gửi lên không hợp lệ.' : data.detail) || message
    } catch {
      /* giữ thông báo mặc định */
    }
    // 401 ở các API thường nghĩa là hết phiên; riêng đăng nhập sai (auth=true) thì để màn hình đăng nhập tự báo lỗi.
    if (response.status === 401 && !auth) unauthorizedHandler?.()
    throw new ApiError(message, response.status)
  }
  return raw ? response : response.json()
}

export const api = {
  authMe: () => request('/api/auth/me', { auth: true }),
  register: (username, password) =>
    request('/api/auth/register', { method: 'POST', json: { username, password }, auth: true }),
  login: (username, password) =>
    request('/api/auth/login', { method: 'POST', json: { username, password }, auth: true }),
  logout: () => request('/api/auth/logout', { method: 'POST', auth: true }),
  changePassword: (current_password, new_password) =>
    request('/api/auth/password', { method: 'POST', json: { current_password, new_password } }),
  session: () => request('/api/session'),
  zaloStatus: () => request('/api/integrations/zalo'),
  configureZaloBot: (token) => request('/api/integrations/zalo/bot', { method: 'PUT', json: { token } }),
  removeZaloBot: () => request('/api/integrations/zalo/bot', { method: 'DELETE' }),
  zaloCode: () => request('/api/integrations/zalo/code', { method: 'POST', json: {} }),
  unlinkZalo: () => request('/api/integrations/zalo', { method: 'DELETE' }),
  chat: (message, signal) => request('/api/chat', { method: 'POST', json: { message }, signal }),
  image: (file, caption, expression, signal, expressionProfile = 'camera') => {
    const form = new FormData()
    form.append('file', file, file.name || 'photo.jpg')
    form.append('caption', caption || '')
    form.append('expression', expression ? 'true' : 'false')
    form.append('expression_profile', expressionProfile)
    return request('/api/image', { method: 'POST', form, signal })
  },
  voice: (blob, filename, signal) => {
    const form = new FormData()
    form.append('file', blob, filename)
    return request('/api/voice', { method: 'POST', form, signal })
  },
  tts: async (text, signal) => {
    const response = await request('/api/tts', { method: 'POST', json: { text }, signal, raw: true })
    return response.blob()
  },
  summarize: ({ text, file }, signal) => {
    const form = new FormData()
    if (file) form.append('file', file, file.name)
    else form.append('text', text || '')
    return request('/api/summarize', { method: 'POST', form, signal })
  },
  getSettings: () => request('/api/settings'),
  saveSettings: (patch) => request('/api/settings', { method: 'PUT', json: patch }),
  testRole: (role) => request('/api/settings/test', { method: 'POST', json: { role } }),
  localResources: () => request('/api/providers/local'),
  messages: () => request('/api/messages'),
  deleteMe: () => request('/api/me', { method: 'DELETE' }),
  clearHistory: () => request('/api/history', { method: 'DELETE' }),
  voiceStatus: () => request('/api/voice/status'),
  timers: () => request('/api/timers'),
  startPomodoro: (work, rest, cycles) =>
    request('/api/timers/pomodoro', {
      method: 'POST',
      json: { work_minutes: work, break_minutes: rest, cycles },
    }),
  setEye: (enabled) => request('/api/timers/eye', { method: 'POST', json: { enabled } }),
  stopTimer: (target) => request(`/api/timers/${target}`, { method: 'DELETE' }),
}
