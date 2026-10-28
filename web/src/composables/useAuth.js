import { reactive } from 'vue'
import { api, onUnauthorized } from './api'

// Trạng thái đăng nhập. required=false nghĩa là máy chủ chạy chế độ ẩn danh (không cần tài khoản).
export const auth = reactive({
  ready: false,
  required: false,
  registration: false,
  user: null,
})

function apply(data) {
  auth.required = data.required
  auth.registration = !!data.registration
  auth.user = data.user || null
  auth.ready = true
}

export async function loadAuth() {
  apply(await api.authMe())
}

export async function login(username, password) {
  apply({ ...(await api.login(username.trim(), password)), registration: auth.registration })
}

export async function register(username, password) {
  apply({ ...(await api.register(username.trim(), password)), registration: auth.registration })
}

export async function logout() {
  try {
    await api.logout()
  } finally {
    auth.user = null
  }
}

export function changePassword(current, next) {
  return api.changePassword(current, next)
}

/** Có thể dùng ứng dụng: đã đăng nhập, hoặc máy chủ không yêu cầu đăng nhập. */
export function isSignedIn() {
  return auth.ready && (!auth.required || !!auth.user)
}

// Hết phiên giữa chừng: quay về màn hình đăng nhập.
onUnauthorized(() => {
  if (auth.required) auth.user = null
})
