<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Copy, KeyRound, Loader, LogOut, ShieldCheck, Trash2, UserRound } from '@lucide/vue'
import { auth, changePassword, logout } from '../composables/useAuth'
import { deleteEverything, messages, newConversation, persistent } from '../composables/useChat'
import { toast } from '../composables/useToast'
import { closeDialog } from '../composables/useUi'
import { api } from '../composables/api'

const current = ref('')
const next = ref('')
const confirm = ref('')
const saving = ref(false)
const zalo = ref(null)
const linkCode = ref(null)
const linking = ref(false)
const botToken = ref('')
const configuringBot = ref(false)
const linkNow = ref(Date.now())
const codeExpired = computed(() => linkCode.value && linkCode.value.expires_at <= linkNow.value)
let linkTimer = null
let alive = true
const sameOwner = (owner) => alive && owner === auth.user
watch(() => auth.user, () => {
  botToken.value = ''
  linkCode.value = null
  zalo.value = null
  if (auth.user) loadZalo()
})

async function loadZalo() {
  if (!auth.required || !auth.user) return
  const owner = auth.user
  try {
    const state = await api.zaloStatus()
    if (!sameOwner(owner)) return
    zalo.value = state
    if (zalo.value.linked) linkCode.value = null
  } catch { /* Keep the last known state during a reconnect. */ }
}

async function createLink() {
  const owner = auth.user
  linking.value = true
  try {
    const code = await api.zaloCode()
    if (sameOwner(owner)) linkCode.value = code
  } catch (error) {
    toast(error.message, { type: 'error' })
  } finally {
    linking.value = false
  }
}

async function copyLinkCommand() {
  if (!linkCode.value || codeExpired.value) return
  try {
    await navigator.clipboard.writeText(`/link ${linkCode.value.code}`)
    toast('Đã sao chép. Dán và gửi lệnh trong chat riêng với bot Zalo.', { type: 'success' })
  } catch {
    toast('Hãy chọn và sao chép nguyên lệnh /link hiển thị bên dưới.', { type: 'error' })
  }
}

async function saveBot() {
  const owner = auth.user
  configuringBot.value = true
  try {
    const state = await api.configureZaloBot(botToken.value.trim())
    if (!sameOwner(owner)) return
    zalo.value = state
    botToken.value = ''
    linkCode.value = null
    if (!zalo.value.linked) {
      const code = await api.zaloCode()
      if (!sameOwner(owner)) return
      linkCode.value = code
    }
    toast(zalo.value.linked ? 'Đã cập nhật bot, liên kết Zalo vẫn được giữ.' : 'Bot đã kết nối. Gửi lệnh /link bên dưới để bật đồng bộ.', { type: 'success' })
  } catch (error) {
    toast(error.message, { type: 'error' })
  } finally {
    configuringBot.value = false
  }
}

async function removeBot() {
  const owner = auth.user
  configuringBot.value = true
  try {
    const state = await api.removeZaloBot()
    if (!sameOwner(owner)) return
    zalo.value = state
    botToken.value = ''
    linkCode.value = null
    toast('Đã xóa cấu hình bot. Lịch sử trên web vẫn được giữ.', { type: 'success' })
  } catch (error) {
    toast(error.message, { type: 'error' })
  } finally {
    configuringBot.value = false
  }
}

async function unlinkZalo() {
  linking.value = true
  try {
    await api.unlinkZalo()
    linkCode.value = null
    await loadZalo()
    toast('Đã hủy liên kết Zalo. Lịch sử trên web vẫn được giữ.', { type: 'success' })
  } catch (error) {
    toast(error.message, { type: 'error' })
  } finally {
    linking.value = false
  }
}

onMounted(() => {
  loadZalo()
  linkTimer = setInterval(() => { linkNow.value = Date.now(); loadZalo() }, 10000)
})
onBeforeUnmount(() => { alive = false; clearInterval(linkTimer); botToken.value = '' })

const canChange = computed(
  () => !saving.value && current.value && next.value.length >= 8 && next.value === confirm.value,
)

async function submitPassword() {
  if (!canChange.value) return
  saving.value = true
  try {
    await changePassword(current.value, next.value)
    current.value = next.value = confirm.value = ''
    toast('Đã đổi mật khẩu. Các thiết bị khác đã bị đăng xuất.', { type: 'success' })
  } catch (error) {
    toast(error.message, { type: 'error' })
  } finally {
    saving.value = false
  }
}

async function signOut() {
  closeDialog()
  await logout()
}

async function clearAll() {
  await newConversation()
  toast('Đã xóa lịch sử hội thoại', { type: 'success', duration: 2500 })
}

async function wipeAll() {
  const message = auth.required
    ? 'Xóa tài khoản và toàn bộ dữ liệu của bạn (hội thoại, cài đặt, API key)? Không thể hoàn tác.'
    : 'Xóa toàn bộ dữ liệu của bạn trên máy chủ (hội thoại, cài đặt, API key)? Không thể hoàn tác.'
  if (!window.confirm(message)) return
  if (await deleteEverything()) {
    closeDialog()
    toast(auth.required ? 'Đã xóa tài khoản' : 'Đã xóa toàn bộ dữ liệu của bạn', { type: 'success', duration: 3000 })
  }
}
</script>

<template>
  <div class="account">
    <section v-if="auth.required && auth.user" class="card who">
      <span class="avatar"><UserRound :size="22" /></span>
      <div>
        <strong>{{ auth.user.username }}</strong>
        <small>Đang đăng nhập</small>
      </div>
      <button class="btn btn-soft" @click="signOut"><LogOut :size="16" /> Đăng xuất</button>
    </section>
    <section v-else class="card note">
      <ShieldCheck :size="18" />
      Máy chủ đang chạy chế độ ẩn danh: dữ liệu gắn với trình duyệt này, không có tài khoản.
    </section>

    <section v-if="auth.required && zalo" class="card">
      <h3><ShieldCheck :size="17" /> Liên kết Zalo</h3>
      <p class="hint">Kết nối bot riêng với tài khoản đang đăng nhập. Token được mã hóa và không hiển thị lại sau khi lưu.</p>
      <p v-if="zalo.bot"><strong>{{ zalo.bot.bot_name }}</strong> · {{ zalo.linked ? 'Đã liên kết và bật đồng bộ' : 'Bot đã kết nối; chờ xác nhận Zalo của bạn' }}</p>
      <p v-else-if="zalo.enabled" class="hint">Bạn đang dùng bot chung của máy chủ. Có thể nhập token ở dưới để chuyển sang bot riêng.</p>
      <form @submit.prevent="saveBot">
        <label for="zalo-bot-token">{{ zalo.bot ? 'Cập nhật token hoặc đổi bot' : 'Token bot Zalo của bạn' }}</label>
        <input id="zalo-bot-token" v-model="botToken" class="field" type="password" autocomplete="new-password" spellcheck="false" placeholder="Dán token từ Zalo Bot Creator" :disabled="configuringBot" />
        <button class="btn btn-primary" type="submit" :disabled="!botToken.trim() || configuringBot || linking">
          <Loader v-if="configuringBot" :size="15" class="spin" /> Lưu và kết nối bot
        </button>
      </form>
      <button v-if="zalo.bot" class="btn btn-soft" :disabled="configuringBot || linking" @click="removeBot">Xóa cấu hình bot của tôi</button>
      <p class="hint">Web và Zalo dùng chung lịch sử, ngữ cảnh trả lời, cài đặt và nhắc nghỉ. Chỉ liên kết với Zalo của bạn; tin nhắn sức khỏe sẽ được chia sẻ giữa hai kênh.</p>
      <p v-if="!zalo.enabled" class="hint">Nhập token ở trên để kết nối bot, không cần sửa cấu hình máy chủ.</p>
      <template v-if="zalo.linked">
        <p>Đã liên kết. Tin nhắn Zalo tự xuất hiện trên web<span v-if="zalo.mirror_web"> và hội thoại web được gửi sang Zalo</span>.</p>
        <button class="btn btn-soft" :disabled="linking" @click="unlinkZalo">Hủy liên kết Zalo</button>
      </template>
      <template v-else-if="zalo.enabled">
        <p class="hint">Chưa đồng bộ tin nhắn. Hoàn tất bước xác nhận: tạo mã rồi gửi nguyên lệnh /link trong chat riêng với bot.</p>
        <button class="btn btn-primary" :disabled="linking || configuringBot" @click="createLink">{{ linkCode ? 'Tạo mã mới' : 'Tạo mã xác nhận Zalo' }}</button>
        <template v-if="linkCode">
          <p>Gửi nguyên lệnh sau trong chat riêng với bot Zalo:</p>
          <code class="link-code">/link {{ linkCode.code }}</code>
          <button class="btn btn-soft" :disabled="codeExpired" @click="copyLinkCommand"><Copy :size="15" /> Sao chép lệnh để gửi trên Zalo</button>
          <small v-if="codeExpired" class="bad">Mã đã hết hạn. Bấm Tạo mã mới rồi gửi lại.</small>
          <small>Mã dùng một lần, hết hạn lúc {{ new Date(linkCode.expires_at).toLocaleTimeString() }}. Không chia sẻ mã với người khác.</small>
        </template>
      </template>
    </section>

    <section v-if="auth.required" class="card">
      <h3><KeyRound :size="17" /> Đổi mật khẩu</h3>
      <form @submit.prevent="submitPassword">
        <input v-model="current" class="field" type="password" autocomplete="current-password" placeholder="Mật khẩu hiện tại" />
        <input v-model="next" class="field" type="password" autocomplete="new-password" placeholder="Mật khẩu mới (ít nhất 8 ký tự)" />
        <input v-model="confirm" class="field" type="password" autocomplete="new-password" placeholder="Nhập lại mật khẩu mới" />
        <small v-if="confirm && confirm !== next" class="bad">Hai mật khẩu chưa khớp.</small>
        <button class="btn btn-primary" type="submit" :disabled="!canChange">
          <Loader v-if="saving" :size="15" class="spin" /> Đổi mật khẩu
        </button>
      </form>
    </section>

    <section class="card">
      <h3><ShieldCheck :size="17" /> Quyền riêng tư</h3>
      <p class="hint">
        <template v-if="persistent">
          Hội thoại, cài đặt và API key (được mã hóa) được lưu trong cơ sở dữ liệu trên máy chủ. Ảnh và âm thanh
          không bao giờ được lưu lại.
        </template>
        <template v-else>Máy chủ đang chạy ở chế độ không lưu trữ: dữ liệu mất khi khởi động lại.</template>
        API key của bạn chỉ dùng để gọi nhà cung cấp bạn chọn; nội dung hội thoại được gửi tới nhà cung cấp đó.
      </p>
      <div class="row">
        <button class="btn btn-danger" :disabled="!messages.length" @click="clearAll"><Trash2 :size="15" /> Xóa lịch sử hội thoại</button>
        <button class="btn btn-danger" @click="wipeAll">
          <Trash2 :size="15" /> {{ auth.required ? 'Xóa tài khoản và dữ liệu' : 'Xóa toàn bộ dữ liệu của tôi' }}
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.account {
  display: grid;
  gap: 14px;
}
.link-code { overflow-wrap: anywhere; user-select: all; padding: 10px; background: var(--surface); border-radius: 8px; }
.card {
  display: grid;
  gap: 10px;
  padding: 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 16px;
}
.who {
  grid-auto-flow: column;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 12px;
}
.who div {
  display: grid;
  line-height: 1.3;
}
.who small {
  color: var(--text-soft);
}
.avatar {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 14px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.note {
  grid-auto-flow: column;
  grid-template-columns: auto 1fr;
  align-items: center;
  font-size: 13.5px;
  color: var(--text-soft);
}
h3 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 15px;
}
form {
  display: grid;
  gap: 8px;
}
form .btn {
  justify-self: end;
}
.bad {
  color: var(--danger);
}
.hint {
  margin: 0;
  font-size: 13px;
  color: var(--text-soft);
}
.row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (max-width: 480px) {
  .who {
    grid-template-columns: auto 1fr;
  }
  .who .btn {
    grid-column: 1 / -1;
  }
}
</style>
