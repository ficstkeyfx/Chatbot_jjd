// A permission prompt may finish after the camera dialog has been closed.
export function createCameraSession(mediaDevices = globalThis.navigator?.mediaDevices) {
  let generation = 0
  let stream = null
  function stop() {
    generation += 1
    stream?.getTracks().forEach((track) => track.stop())
    stream = null
  }
  async function open(facingMode = 'user') {
    stop()
    const request = generation
    if (!mediaDevices?.getUserMedia) throw new Error('CameraUnavailable')
    const incoming = await mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 960 } },
    })
    if (request !== generation) {
      incoming.getTracks().forEach((track) => track.stop())
      return null
    }
    stream = incoming
    return stream
  }
  return { open, stop }
}
