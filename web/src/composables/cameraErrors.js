export function cameraErrorMessage(cause, { secure = true, policyAllowed = true, permission = 'unknown' } = {}) {
  if (!secure) return 'Trang chưa được trình duyệt coi là an toàn. Mở http://localhost:18000 hoặc HTTPS có chứng chỉ hợp lệ.'
  if (!policyAllowed) return 'Camera bị chặn bởi chính sách của trang hoặc khung nhúng. Hãy mở web trực tiếp trong một tab Chrome/Edge.'
  if (cause?.name === 'NotFoundError') return 'Không tìm thấy camera trên thiết bị này.'
  if (cause?.name === 'NotReadableError') return 'Không đọc được camera. Hãy đóng ứng dụng khác đang dùng camera và kiểm tra công tắc camera trên laptop.'
  if (cause?.name === 'NotAllowedError' || cause?.name === 'SecurityError') {
    if (permission === 'denied') return 'Trình duyệt đang chặn camera cho địa chỉ này. Vào Cài đặt trang web → Camera → Cho phép, rồi tải lại trang.'
    return 'Trình duyệt hoặc hệ điều hành từ chối camera dù chưa hỏi quyền. Kiểm tra chứng chỉ HTTPS có được Chrome tin cậy, quyền camera của Windows và thử http://localhost:18000.'
  }
  return 'Không mở được camera. Hãy thử lại hoặc chọn ảnh từ thiết bị.'
}
