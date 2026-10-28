import { reactive } from 'vue'

export const toasts = reactive([])
let nextId = 1

export function toast(message, { type = 'info', duration = 4500, title = '' } = {}) {
  const id = nextId++
  toasts.push({ id, message, type, title })
  if (duration > 0) setTimeout(() => dismissToast(id), duration)
  return id
}

export function dismissToast(id) {
  const index = toasts.findIndex((item) => item.id === id)
  if (index !== -1) toasts.splice(index, 1)
}
