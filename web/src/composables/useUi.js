import { reactive, watchEffect } from 'vue'

const THEME_KEY = 'hb.theme'

function readTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* bỏ qua khi bị chặn lưu trữ */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

// Trạng thái giao diện dùng chung: ngăn kéo, hộp thoại, chủ đề.
export const ui = reactive({
  theme: readTheme(),
  drawer: false,
  dialog: null, // 'settings' | 'summary' | null
  pick: { n: 0, expression: false }, // yêu cầu mở hộp chọn ảnh từ nơi khác (thanh bên, lệnh chat)
})

watchEffect(() => {
  document.documentElement.dataset.theme = ui.theme
  try {
    localStorage.setItem(THEME_KEY, ui.theme)
  } catch {
    /* bỏ qua */
  }
})

export function toggleTheme() {
  ui.theme = ui.theme === 'dark' ? 'light' : 'dark'
}

export function openDialog(name) {
  ui.dialog = name
  ui.drawer = false
}

export function closeDialog() {
  ui.dialog = null
}

export function requestPick(expression = false) {
  ui.drawer = false
  ui.pick = { n: ui.pick.n + 1, expression }
}
