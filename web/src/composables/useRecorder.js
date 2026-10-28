import { computed, onBeforeUnmount, ref } from 'vue'

const MAX_SECONDS = 60

function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return ''
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || ''
}

/** Ghi âm bằng MediaRecorder; trả về Blob khi người dùng dừng, không gửi gì khi huỷ. */
export function useRecorder() {
  const recording = ref(false)
  const seconds = ref(0)
  const supported = computed(
    () => !!(navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'),
  )

  let recorder = null
  let stream = null
  let chunks = []
  let timer = null
  let resolveResult = null
  let cancelled = false

  function cleanup() {
    clearInterval(timer)
    timer = null
    stream?.getTracks().forEach((track) => track.stop())
    stream = null
    recording.value = false
  }

  async function start() {
    if (recording.value) return
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    })
    const mimeType = pickMimeType()
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    chunks = []
    cancelled = false
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data)
    }
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || 'audio/webm'
      const blob = cancelled || !chunks.length ? null : new Blob(chunks, { type })
      cleanup()
      resolveResult?.(blob)
      resolveResult = null
    }
    recorder.start()
    seconds.value = 0
    recording.value = true
    timer = setInterval(() => {
      seconds.value += 1
      if (seconds.value >= MAX_SECONDS) stop()
    }, 1000)
  }

  function stop() {
    return new Promise((resolve) => {
      if (!recorder || recorder.state === 'inactive') return resolve(null)
      resolveResult = resolve
      recorder.stop()
    })
  }

  function cancel() {
    cancelled = true
    if (recorder && recorder.state !== 'inactive') recorder.stop()
    else cleanup()
  }

  onBeforeUnmount(cancel)

  return { recording, seconds, supported, start, stop, cancel, maxSeconds: MAX_SECONDS }
}
