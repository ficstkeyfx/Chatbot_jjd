<script setup>
import { computed, ref } from 'vue'
import { Eye, EyeOff, Heart, Loader, LogIn, ShieldCheck, UserPlus } from '@lucide/vue'
import { auth, login, register } from '../composables/useAuth'

const mode = ref('login') // 'login' | 'register'
const username = ref('')
const password = ref('')
const confirm = ref('')
const showPassword = ref(false)
const busy = ref(false)
const error = ref('')

const isRegister = computed(() => mode.value === 'register')
const canSubmit = computed(
  () =>
    !busy.value &&
    username.value.trim().length >= 3 &&
    password.value.length >= (isRegister.value ? 8 : 1) &&
    (!isRegister.value || confirm.value === password.value),
)
const mismatch = computed(() => isRegister.value && confirm.value && confirm.value !== password.value)

function switchMode(next) {
  mode.value = next
  error.value = ''
  confirm.value = ''
}

async function submit() {
  if (!canSubmit.value) return
  busy.value = true
  error.value = ''
  try {
    await (isRegister.value ? register : login)(username.value, password.value)
    password.value = confirm.value = ''
  } catch (failure) {
    error.value = failure.message
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <main class="auth">
    <section class="card" aria-labelledby="auth-title">
      <span class="logo"><Heart :size="26" fill="currentColor" /></span>
      <h1 id="auth-title">Trợ lý Sức khỏe</h1>
      <p class="sub">Giảm kiệt quệ kỹ thuật số: hỏi đáp, giọng nói, tóm tắt bài học, nghỉ mắt và Pomodoro.</p>

      <div v-if="auth.registration" class="tabs" role="tablist">
        <button role="tab" :aria-selected="!isRegister" :class="{ on: !isRegister }" @click="switchMode('login')">
          Đăng nhập
        </button>
        <button role="tab" :aria-selected="isRegister" :class="{ on: isRegister }" @click="switchMode('register')">
          Tạo tài khoản
        </button>
      </div>

      <form @submit.prevent="submit" novalidate>
        <label>
          Tên đăng nhập
          <input
            v-model="username"
            class="field"
            autocomplete="username"
            autocapitalize="none"
            spellcheck="false"
            maxlength="32"
            :placeholder="isRegister ? '3-32 ký tự: chữ không dấu, số, . _ -' : ''"
            autofocus
          />
        </label>
        <label>
          Mật khẩu
          <span class="pw">
            <input
              v-model="password"
              class="field"
              :type="showPassword ? 'text' : 'password'"
              :autocomplete="isRegister ? 'new-password' : 'current-password'"
              :placeholder="isRegister ? 'Ít nhất 8 ký tự' : ''"
              maxlength="200"
            />
            <button type="button" class="icon-btn" :aria-label="showPassword ? 'Ẩn mật khẩu' : 'Hiện mật khẩu'" @click="showPassword = !showPassword">
              <EyeOff v-if="showPassword" :size="18" />
              <Eye v-else :size="18" />
            </button>
          </span>
        </label>
        <label v-if="isRegister">
          Nhập lại mật khẩu
          <input v-model="confirm" class="field" :type="showPassword ? 'text' : 'password'" autocomplete="new-password" maxlength="200" />
          <small v-if="mismatch" class="bad">Hai mật khẩu chưa khớp.</small>
        </label>

        <p v-if="error" class="error" role="alert">{{ error }}</p>

        <button class="btn btn-primary submit" type="submit" :disabled="!canSubmit">
          <Loader v-if="busy" :size="17" class="spin" />
          <UserPlus v-else-if="isRegister" :size="17" />
          <LogIn v-else :size="17" />
          {{ isRegister ? 'Tạo tài khoản' : 'Đăng nhập' }}
        </button>
      </form>

      <p v-if="!auth.registration" class="hint">Đăng ký đang đóng. Hãy liên hệ quản trị viên để được cấp tài khoản.</p>
      <p class="note">
        <ShieldCheck :size="15" />
        <span>
          Mỗi tài khoản có API key, mô hình và dữ liệu hội thoại riêng. Chỉ hỗ trợ ban đầu, không thay thế bác sĩ;
          khẩn cấp hãy gọi <a href="tel:115">115</a>.
        </span>
      </p>
    </section>
  </main>
</template>

<style scoped>
.auth {
  min-height: 100%;
  display: grid;
  place-items: center;
  padding: 24px 16px;
  overflow-y: auto;
  background:
    radial-gradient(800px 420px at 80% -10%, color-mix(in srgb, var(--primary) 16%, transparent), transparent),
    radial-gradient(700px 400px at -10% 110%, color-mix(in srgb, var(--accent) 12%, transparent), transparent),
    var(--bg);
}
.card {
  width: min(420px, 100%);
  display: grid;
  gap: 14px;
  padding: 30px 28px 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: var(--shadow-lg);
  text-align: center;
}
.logo {
  justify-self: center;
  display: grid;
  place-items: center;
  width: 60px;
  height: 60px;
  border-radius: 20px;
  color: #fff;
  background: linear-gradient(135deg, var(--primary), #3fd0b0);
  box-shadow: 0 10px 26px rgba(15, 157, 131, 0.35);
}
h1 {
  margin: 4px 0 0;
  font-size: 24px;
}
.sub {
  margin: 0;
  color: var(--text-soft);
  font-size: 14px;
}
.tabs {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--bg-soft);
  border-radius: 12px;
}
.tabs button {
  flex: 1;
  padding: 8px;
  border-radius: 9px;
  font-weight: 600;
  color: var(--text-soft);
}
.tabs button.on {
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-sm);
}
form {
  display: grid;
  gap: 12px;
  text-align: left;
}
label {
  display: grid;
  gap: 5px;
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text-soft);
}
.pw {
  display: flex;
  align-items: center;
  gap: 4px;
}
.bad,
.error {
  color: var(--danger);
  font-weight: 500;
}
.error {
  margin: 0;
  padding: 9px 12px;
  border-radius: 10px;
  background: var(--danger-soft);
  font-size: 13.5px;
}
.submit {
  margin-top: 4px;
  padding: 12px;
}
.hint {
  margin: 0;
  font-size: 13px;
  color: var(--text-soft);
}
.note {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  margin: 0;
  text-align: left;
  font-size: 12.5px;
  color: var(--muted);
}
.note svg {
  flex: none;
  margin-top: 2px;
  color: var(--primary);
}
.note a {
  color: var(--danger);
  font-weight: 700;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
