// Chuyển văn bản của bot thành các khối có cấu trúc (đoạn, danh sách, in đậm).
// Không dùng v-html: mọi nội dung đều được render bằng template nên không có nguy cơ XSS.

const BULLET = /^\s*(?:[-*•+])\s+(.*)$/
const NUMBERED = /^\s*(\d{1,2})[.)]\s+(.*)$/

function inline(text) {
  const parts = []
  const pattern = /\*\*(.+?)\*\*|__(.+?)__/g
  let last = 0
  let match
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push({ text: text.slice(last, match.index) })
    parts.push({ text: match[1] ?? match[2], bold: true })
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push({ text: text.slice(last) })
  return parts.length ? parts : [{ text }]
}

export function formatText(source) {
  const blocks = []
  let list = null
  const flush = () => {
    if (list) blocks.push(list)
    list = null
  }
  for (const rawLine of String(source ?? '').replace(/\r\n?/g, '\n').split('\n')) {
    const line = rawLine.replace(/^#{1,6}\s+/, '')
    if (!line.trim()) {
      flush()
      continue
    }
    const bullet = BULLET.exec(line)
    const numbered = NUMBERED.exec(line)
    if (bullet || numbered) {
      const type = numbered ? 'ol' : 'ul'
      if (!list || list.type !== type) {
        flush()
        list = { type, items: [] }
      }
      list.items.push(inline(bullet ? bullet[1] : numbered[2]))
      continue
    }
    flush()
    blocks.push({ type: 'p', parts: inline(line.trim()) })
  }
  flush()
  return blocks
}

export function formatClock(timestamp) {
  return new Date(timestamp).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
}

export function formatMmSs(totalSeconds) {
  const safe = Math.max(0, Math.round(totalSeconds))
  const minutes = Math.floor(safe / 60)
  const seconds = safe % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}
